import os
import random
import argparse
from typing import List, Tuple

def parse_lines_file(lines_file_path: str, images_dir: str) -> List[Tuple[str, str]]:
    """
    Parse the lines.txt file to extract image paths and corresponding text labels.

    Args:
        lines_file_path (str): Path to the lines.txt file.
        images_dir (str): Directory containing the image files.

    Returns:
        List[Tuple[str, str]]: List containing image paths and text labels.
    """
    data = []
    with open(lines_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue  # Skip comments and empty lines
            parts = line.split(' ')
            if len(parts) < 9:
                continue  # Skip incorrectly formatted lines
            line_id = parts[0]
            status = parts[1]
            if status != 'ok':
                continue  # Skip lines marked as 'err'
            transcription = ' '.join(parts[8:]).replace('|', ' ')
            # Construct image file path
            form_id = line_id.split('-')[0]
            page_id = '-'.join(line_id.split('-')[:2])
            image_filename = f"{line_id}.png"
            image_path = os.path.join(images_dir, form_id, page_id, image_filename)
            if not os.path.exists(image_path):
                print(f"Warning: Image file does not exist: {image_path}")
                continue
            data.append((image_path, transcription))
    return data

def split_dataset(data: List[Tuple[str, str]], train_ratio: float, val_ratio: float, seed: int = 42):
    """
    Split the data into training, validation, and test sets according to the specified ratios.

    Args:
        data (List[Tuple[str, str]]): Original data list.
        train_ratio (float): Training set ratio.
        val_ratio (float): Validation set ratio.
        seed (int): Random seed.

    Returns:
        Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]: Training, validation, and test sets.
    """
    random.seed(seed)
    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data

def save_labels(data: List[Tuple[str, str]], output_path: str):
    """
    Save the data in PaddleOCR's required label format.

    Args:
        data (List[Tuple[str, str]]): Data list.
        output_path (str): Output file path.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for image_path, transcription in data:
            f.write(f"{image_path}\t{transcription}\n")

def main():
    parser = argparse.ArgumentParser(description="Convert IAM lines.txt to PaddleOCR text recognition training format and split the dataset.")
    parser.add_argument('--lines_file', type=str, required=True, help='Path to lines.txt file')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory containing image files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output label files')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Parsing lines.txt file...")
    data = parse_lines_file(args.lines_file, args.images_dir)
    print(f"Total valid data entries: {len(data)}")

    print("Splitting dataset...")
    train_data, val_data, test_data = split_dataset(data, args.train_ratio, args.val_ratio, args.seed)
    print(f"Training set: {len(train_data)} entries")
    print(f"Validation set: {len(val_data)} entries")
    print(f"Test set: {len(test_data)} entries")

    print("Saving label files...")
    save_labels(train_data, os.path.join(args.output_dir, 'train.txt'))
    save_labels(val_data, os.path.join(args.output_dir, 'val.txt'))
    save_labels(test_data, os.path.join(args.output_dir, 'test.txt'))
    print("All label files have been saved.")

if __name__ == '__main__':
    main()
