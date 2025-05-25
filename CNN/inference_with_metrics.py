import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from typing import List
from jiwer import cer, wer  # <-- added here

from train import (
    HandwritingDataset,
    pad_collate,
    build_vocab,
    CRNN,
    greedy_decode,
)

def run_inference(
    model_path: str,
    data_root: str,
    label_file: str,
    batch_size: int = 32,
    output_file: str = "inference_results.txt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build vocabulary
    char2idx, idx2char = build_vocab(label_file)
    test_dataset = HandwritingDataset(data_root, label_file, char2idx)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: pad_collate(b, len(char2idx), 2 * 2 * 1),
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    # Load model
    model = CRNN(num_classes=len(char2idx))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    predictions: List[str] = []
    ground_truths: List[str] = []

    with torch.no_grad():
        for images, labels, input_lens, label_lens in test_loader:
            images = images.to(device)
            input_lens = input_lens.to(device)

            # Predict
            pred_texts = greedy_decode(model, images, input_lens, idx2char)
            true_texts = [
                "".join(idx2char.get(int(i), "") for i in lbl[:ll])
                for lbl, ll in zip(labels, label_lens)
            ]

            predictions.extend(pred_texts)
            ground_truths.extend(true_texts)

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        for gt, pred in zip(ground_truths, predictions):
            f.write(f"GT:\t{gt}\nPR:\t{pred}\n\n")

    # Compute CER and WER
    cer_score = cer(ground_truths, predictions)
    wer_score = wer(ground_truths, predictions)

    print(f"\nInference complete. Results saved to {output_file}")
    print(f"Character Error Rate (CER): {cer_score:.4f}")
    print(f"Word Error Rate (WER):     {wer_score:.4f}")

if __name__ == "__main__":
    run_inference(
        model_path="/data/haozhen/Irene/crnn_ctc_pytorch_30.pth",
        data_root="/data/haozhen/Irene",  # absolute path to train_data directory
        label_file="/data/haozhen/Irene/train_data/rec/rec_gt_test.txt",  # absolute path to train_data/rec/rec_gt_test.txt
    )
