
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
import pandas as pd

def clean_data(df):
    """Verify image paths, then remove rows with missing or empty labels."""
    print("Verifying image paths...")
    valid_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        rel = row['image_path']
        img_path = os.path.join('data/lines', rel[:3], rel[:7], rel + '.png')
        try:
            with Image.open(img_path) as img:
                img.verify()
            valid_data.append(row)
        except (FileNotFoundError, IOError):
            continue

    df_valid = pd.DataFrame(valid_data)
    print(f"Valid images: {len(df_valid)} / {len(df)}")

    # Log how many labels are null or blank
    null_count = df_valid['label'].isnull().sum()
    blank_count = (df_valid['label'].str.strip() == '').sum()
    print(f"Null labels: {null_count}")
    print(f"Blank labels: {blank_count}")

    # Remove missing/blank labels
    before = len(df_valid)
    df_valid = df_valid[df_valid['label'].notnull()]
    df_valid = df_valid[df_valid['label'].str.strip() != '']
    print(f"Final cleaned: {len(df_valid)} / {before}")

    return df_valid
