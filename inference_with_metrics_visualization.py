import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List
from jiwer import cer, wer  # <-- added here
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import math
import torch.nn as nn
import torch.nn.functional as F
from train import (
    build_vocab,
    CRNN,
    greedy_decode,
)

class HandwritingDataset(Dataset):
    """Loads (image, transcription) pairs listed in a TSV file.

    Each line in *label_file* should be:  <img_rel_path>\t<label>\n
    Images are resized to fixed *target_height* while preserving aspect ratio.
    """

    def __init__(
        self,
        data_root: str | Path,
        label_file: str | Path,
        char2idx: dict[str, int],
        target_height: int = 64,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.samples: List[Tuple[Path, str]] = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                img_rel, label = line.rstrip("\n").split("\t", 1)
                self.samples.append((self.data_root / img_rel, label))
        self.char2idx = char2idx
        self.blank_idx = len(char2idx)  # final slot reserved for blank
        self.to_tensor = transforms.ToTensor()  # converts HWC [0,255] -> CHW [0,1]
        self.target_height = target_height

    def _text_to_indices(self, text: str) -> List[int]:
        # Replace '|' with space (same logic as TF script)
        return [self.char2idx.get(ch, self.blank_idx) for ch in text.replace("|", " ")]

    def _resize_keep_ratio(self, img: Image.Image) -> Image.Image:
        if img.height == self.target_height:
            return img
        ratio = self.target_height / img.height
        new_w = max(1, int(img.width * ratio))
        return img.resize((new_w, self.target_height), Image.Resampling.LANCZOS)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # grayscale
        img = self._resize_keep_ratio(img)
        img_tensor = self.to_tensor(img)  # (1, H, W) float32 in [0,1]
        seq = torch.tensor(self._text_to_indices(label), dtype=torch.long)
        return img_tensor, seq, str(img_path)
    
def pad_collate(batch, blank_idx: int, downsample_factor: int):
    """Pads images & labels in a batch, returning tensors required by CTC."""
    imgs, labels, paths = zip(*batch)

    # --- pad images to max width ---
    heights = [im.shape[1] for im in imgs]
    assert len(set(heights)) == 1, "all images must share the target height"
    max_w = max(im.shape[2] for im in imgs)
    padded_imgs = []
    for im in imgs:
        pad_w = max_w - im.shape[2]
        padded = F.pad(im, (0, pad_w, 0, 0), value=1.0)  # right‑pad with white
        padded_imgs.append(padded)
    image_batch = torch.stack(padded_imgs)  # (B, 1, H, W)

    # --- pad label sequences ---
    label_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
    max_lab = max(label_lens)
    padded_labels = torch.full((len(labels), max_lab), blank_idx, dtype=torch.long)
    for i, seq in enumerate(labels):
        padded_labels[i, : len(seq)] = seq

    # CTC expects (T, B, C); we also need input_lengths after CNN down‑sampling
    input_lens = torch.full(
        (len(labels),),
        fill_value=math.floor(max_w / downsample_factor),
        dtype=torch.long,
    )

    return image_batch, padded_labels, input_lens, label_lens, paths

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

    n_plots = 10  # <-- set how many samples you want to save
    saved = 0
    os.makedirs("output", exist_ok=True)

    with torch.no_grad():
        for images, labels, input_lens, label_lens, paths in test_loader:
            images = images.to(device)
            input_lens = input_lens.to(device)

            pred_texts = greedy_decode(model, images, input_lens, idx2char)
            true_texts = [
                "".join(idx2char.get(int(i), "") for i in lbl[:ll])
                for lbl, ll in zip(labels, label_lens)
            ]

            predictions.extend(pred_texts)
            ground_truths.extend(true_texts)

            for path, gt, pred in zip(paths, true_texts, pred_texts):
                if saved >= n_plots:
                    break

                # Load image
                img = Image.open(path).convert("L")

                # Create plot
                plt.figure(figsize=(8, 3))
                plt.imshow(img, cmap="gray")
                plt.title(f"GT: {gt}\nPR: {pred}")
                plt.axis("off")

                # Use relative path name for the image
                fname = os.path.basename(path).replace("/", "_")
                save_path = os.path.join("output", f"pred_{fname}.png")
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()

                saved += 1
            if saved >= n_plots:
                break

    # Save results
    # with open(output_file, "w", encoding="utf-8") as f:
    #     for gt, pred in zip(ground_truths, predictions):
    #         f.write(f"GT:\t{gt}\nPR:\t{pred}\n\n")

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
