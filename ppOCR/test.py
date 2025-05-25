import os
import json
import subprocess
from Levenshtein import distance
from tqdm import tqdm

def calculate_cer(pred_text, gt_text):
    """
    Calculate Character Error Rate (CER)
    CER = (substitutions + insertions + deletions) / number of characters in ground truth
    """
    if not gt_text:
        return 1.0 if pred_text else 0.0
    
    # Calculate edit distance (substitutions + insertions + deletions)
    edit_distance = distance(pred_text, gt_text)
    
    # Divide by number of characters in ground truth
    return edit_distance / len(gt_text)

def calculate_wer(pred_text, gt_text):
    """
    Calculate Word Error Rate (WER)
    WER = (substitutions + insertions + deletions) / number of words in ground truth
    """
    # Split into words and handle empty cases
    pred_words = pred_text.split()
    gt_words = gt_text.split()
    
    if not gt_words:
        return 1.0 if pred_words else 0.0
    
    # Calculate edit distance at word level
    edit_distance = distance(pred_words, gt_words)
    
    # Divide by number of words in ground truth
    return edit_distance / len(gt_words)

def run_ocr_inference(image_path):
    """Run OCR inference using PaddleOCR command line"""
    # Get the absolute path to the PaddleOCR directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    paddle_ocr_dir = os.path.join(current_dir, 'PaddleOCR-release-2.8')
    
    # Change to PaddleOCR directory
    original_dir = os.getcwd()
    os.chdir(paddle_ocr_dir)
    
    try:
        # Convert to forward slashes and make path relative to PaddleOCR directory
        rel_image_path = os.path.relpath(image_path, paddle_ocr_dir).replace('\\', '/')
        
        # Run PaddleOCR command
        cmd = [
            'python', 'tools/infer_rec.py',
            '-c', 'configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml',
            '-o', f'Global.infer_img={rel_image_path}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse the output to get the prediction
        output_lines = result.stdout.strip().split('\n')
        prediction = None
        
        for line in output_lines:
            if 'result:' in line:
                parts = line.split('result:')[1].strip().split(',')
                if parts:
                    prediction = parts[0].strip()
                    break
        
        if not prediction:
            return ""
            
        return prediction
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return ""
    finally:
        # Change back to original directory
        os.chdir(original_dir)

def convert_path(path):
    """Convert path from test.txt format to actual directory structure"""
    # Convert Windows backslashes to forward slashes
    return path.replace('\\', '/')

def find_image(base_dir, target_filename):
    """Find an image in the directory structure by filename"""
    for root, dirs, files in os.walk(base_dir):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None

def load_ground_truth(gt_file):
    """Load ground truth from test.txt format"""
    ground_truth = {}
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Split on tab character
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # Convert path to correct format
                    image_path = convert_path(parts[0])
                    gt_text = parts[1]
                    ground_truth[image_path] = gt_text
        return ground_truth
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_file}")
        print("Current working directory:", os.getcwd())
        print("Available files in pplabels directory:")
        try:
            print(os.listdir('pplabels'))
        except:
            print("Could not list pplabels directory")
        raise

def main():
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lines_dir = os.path.join(current_dir, 'lines')
    gt_file = os.path.join(current_dir, 'pplabels', 'val.txt')
    
    # Load ground truth
    ground_truth = load_ground_truth(gt_file)
    print(f"Loaded {len(ground_truth)} ground truth entries")
    
    # Take first 1000 entries
    ground_truth = dict(list(ground_truth.items())[:1000])
    print(f"Selected {len(ground_truth)} entries for testing")
    
    # Initialize metrics
    total_cer = 0
    total_wer = 0
    num_samples = 0
    predictions = {}
    
    # Process each test image
    for image_path, gt_text in tqdm(ground_truth.items(), desc="Processing images"):
        # Convert relative path to absolute path
        abs_image_path = os.path.join(current_dir, image_path)
        
        if not os.path.exists(abs_image_path):
            continue
            
        # Get prediction using command line interface
        pred_text = run_ocr_inference(abs_image_path)
        
        if not pred_text:
            continue
            
        # Calculate metrics
        cer = calculate_cer(pred_text, gt_text)
        wer = calculate_wer(pred_text, gt_text)
        
        total_cer += cer
        total_wer += wer
        num_samples += 1
        
        # Store prediction with detailed metrics
        predictions[image_path] = {
            'prediction': pred_text,
            'ground_truth': gt_text,
            'cer': cer,
            'wer': wer,
            'cer_details': {
                'edit_distance': distance(pred_text, gt_text),
                'gt_length': len(gt_text)
            },
            'wer_details': {
                'edit_distance': distance(pred_text.split(), gt_text.split()),
                'gt_word_count': len(gt_text.split())
            }
        }
    
    # Calculate average metrics
    avg_cer = total_cer / num_samples if num_samples > 0 else 0
    avg_wer = total_wer / num_samples if num_samples > 0 else 0
    
    # Save predictions and metrics
    results = {
        'metrics': {
            'average_cer': avg_cer,
            'average_wer': avg_wer,
            'num_samples': num_samples
        },
        'predictions': predictions
    }
    
    # Save detailed results to JSON
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save summary to text file
    with open('evaluation_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"=================\n")
        f.write(f"Total images processed: {num_samples}\n")
        f.write(f"Average CER: {avg_cer:.4f}\n")
        f.write(f"Average WER: {avg_wer:.4f}\n\n")
        
        f.write("Detailed Results\n")
        f.write("===============\n")
        for path, result in predictions.items():
            f.write(f"\nImage: {path}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"Ground Truth: {result['ground_truth']}\n")
            f.write(f"CER: {result['cer']:.4f}\n")
            f.write(f"WER: {result['wer']:.4f}\n")
    
    print(f"\nEvaluation complete!")
    print(f"Total images processed: {num_samples}")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Detailed results saved to evaluation_results.json")
    print(f"Summary saved to evaluation_summary.txt")

if __name__ == "__main__":
    main() 