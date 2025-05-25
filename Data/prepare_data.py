
import os
import tarfile
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class HandwritingDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, char_to_idx=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.char_to_idx = char_to_idx

        if self.char_to_idx is None:
            raise ValueError("char_to_idx must be provided to encode labels.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel = self.image_paths[idx]
        img_path = os.path.join('data/lines', rel[:3], rel[:7], rel + '.png')
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        label_str = self.labels[idx]
        encoded_label = torch.tensor([self.char_to_idx[c] for c in label_str], dtype=torch.long)

        return image, encoded_label

def extract_tgz(tgz_path, extract_path):
    print(f"Extracting {tgz_path} to {extract_path}...")
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

def process_labels(lines_txt_path):
    print("Processing labels...")
    data = []
    with open(lines_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 9:
                image_path = parts[0]
                label = ' '.join(parts[8:]).replace('|', ' ').replace('#', '')
                data.append({'image_path': image_path, 'label': label})
    return pd.DataFrame(data)

def create_char_mapping(labels):
    chars = set(''.join(labels))
    char_list = sorted(chars)
    char_to_idx = {char: idx + 1 for idx, char in enumerate(char_list)}  # Start at 1
    char_to_idx['<blank>'] = 0  # 0 = CTC blank token
    return char_to_idx

def prepare_data():
    os.makedirs('data/lines', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    if not os.path.exists('data/lines/lines'):
        extract_tgz('lines.tgz', 'data/lines')

    df = process_labels('lines.txt')
    char_to_idx = create_char_mapping(df['label'])
    np.save('data/processed/char_to_idx.npy', char_to_idx)

    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    np.random.seed(42)
    indices = np.random.permutation(len(df))
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = HandwritingDataset(
        df.iloc[train_indices]['image_path'].values,
        df.iloc[train_indices]['label'].values,
        transform=transform,
        char_to_idx=char_to_idx
    )

    val_dataset = HandwritingDataset(
        df.iloc[val_indices]['image_path'].values,
        df.iloc[val_indices]['label'].values,
        transform=transform,
        char_to_idx=char_to_idx
    )

    test_dataset = HandwritingDataset(
        df.iloc[test_indices]['image_path'].values,
        df.iloc[test_indices]['label'].values,
        transform=transform,
        char_to_idx=char_to_idx
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(" Data preparation completed!")
    print(f" Training samples: {len(train_dataset)}")
    print(f" Validation samples: {len(val_dataset)}")
    print(f" Test samples: {len(test_dataset)}")
    print(f" Unique characters (incl. <blank>): {len(char_to_idx)}")

    return train_loader, val_loader, test_loader, char_to_idx

if __name__ == "__main__":
    prepare_data()
