from jiwer import wer, cer
import numpy as np

# Load ground truth
gt = []
with open('train_data/rec/rec_gt_test.txt', 'r') as f:
    for line in f:
        _, label = line.strip().split('\t')
        gt.append(label.replace('|', ' '))

# Dummy predictions (replace this with your model outputs)
# Here assuming same order as gt for simplicity
pred = [...]  # a list of predicted strings matching gt length

# Compute metrics
char_err = cer(gt, pred)
word_err = wer(gt, pred)
seq_acc = np.mean([g == p for g, p in zip(gt, pred)])

print(f"CER: {char_err:.4f}")
print(f"WER: {word_err:.4f}")
print(f"Sequence Accuracy: {seq_acc:.4f}")
