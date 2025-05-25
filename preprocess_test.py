import os
import random
import shutil

# Paths
base_dir = 'train_data/rec'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

label_file = os.path.join(base_dir, 'rec_gt_train.txt')
test_label_file = os.path.join(base_dir, 'rec_gt_test.txt')

# Load all lines
with open(label_file, 'r') as f:
    lines = f.readlines()

# Shuffle and split
random.seed(42)  # for reproducibility
random.shuffle(lines)
num_test = int(0.3 * len(lines))
test_lines = lines[:num_test]
train_lines = lines[num_test:]

# Move image files to test directory and adjust path
new_test_lines = []
for line in test_lines:
    image_path, label = line.strip().split('\t')
    image_name = os.path.basename(image_path)
    src_path = os.path.join(train_dir, image_name)
    dst_path = os.path.join(test_dir, image_name)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
    new_test_lines.append(f"{os.path.join(base_dir, 'test', image_name)}\t{label}\n")

# Save new label files
with open(test_label_file, 'w') as f:
    f.writelines(new_test_lines)

with open(label_file, 'w') as f:
    f.writelines(train_lines)

print(f"Moved {len(test_lines)} samples to test split.")
