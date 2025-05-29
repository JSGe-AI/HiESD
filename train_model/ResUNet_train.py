import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
from collections import defaultdict # Use defaultdict for easier accumulation

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define number of classes (background + 4 objects)
NUM_CLASSES = 5
# Define foreground classes (assuming 0 is background)
FOREGROUND_CLASSES = list(range(1, NUM_CLASSES)) # Classes 1, 2, 3, 4

class SegmentationDataset(Dataset):
    def __init__(self, image_txt_path, transform=None):
        self.image_paths = []
        self.mask_paths = []
        with open(image_txt_path, 'r') as f:
            for line in f:
                image_path = line.strip()
                self.image_paths.append(image_path)
                mask_path = image_path.replace("patch_512", "mask_4cls")
                self.mask_paths.append(mask_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L") # Grayscale mask
        if self.transform:
            mask_np = np.array(mask)
            transformed = self.transform(image=np.array(image), mask=mask_np)
            image = transformed["image"]
            mask = transformed["mask"].long()
        return image, mask


# Data preprocessing and augmentation (No changes needed here)
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# --- Metric Calculation Functions ---
def calculate_batch_metrics(preds, labels, num_classes, smooth=1e-7):
    """
    Calculates per-class Precision, Recall, Dice score, IoU FOR A SINGLE BATCH.
    Crucially, it also identifies which foreground classes are present in the labels.

    Args:
        preds (torch.Tensor): Model output logits (B, C, H, W).
        labels (torch.Tensor): Ground truth labels (B, H, W).
        num_classes (int): Total number of classes (including background).
        smooth (float): Smoothing factor for metric calculation.

    Returns:
        tuple: (metrics_dict, present_classes_set)
            metrics_dict (dict): Contains 'per_class_precision', 'per_class_recall',
                                 'per_class_dice', 'per_class_iou' for the batch.
                                 Values are computed for ALL foreground classes,
                                 relying on 'smooth' for stability even if absent.
            present_classes_set (set): A set of foreground class indices (int)
                                        that are actually present in the `labels`
                                        of this batch.
    """
    preds_argmax = preds.argmax(dim=1) # (B, H, W)

    per_class_precision = {}
    per_class_recall = {}
    per_class_dice = {}
    per_class_iou = {}

    # Identify foreground classes present in this batch's labels
    unique_labels = torch.unique(labels)
    present_foreground_classes = set(cls_idx for cls_idx in unique_labels.cpu().numpy() if cls_idx in FOREGROUND_CLASSES)

    # Calculate metrics for all potential foreground classes
    for cls_idx in FOREGROUND_CLASSES:
        pred_mask = (preds_argmax == cls_idx)
        true_mask = (labels == cls_idx)

        tp = (pred_mask & true_mask).sum().float()
        fp = (pred_mask & ~true_mask).sum().float()
        fn = (~pred_mask & true_mask).sum().float()

        # --- Precision Calculation ---
        precision = tp / (tp + fp + smooth)
        per_class_precision[cls_idx] = precision.item()

        # --- Recall Calculation ---
        recall = tp / (tp + fn + smooth)
        per_class_recall[cls_idx] = recall.item()

        # --- Dice Score Calculation ---
        dice = (2.0 * tp + smooth) / (tp + fp + tp + fn + smooth)
        per_class_dice[cls_idx] = dice.item()

        # --- IoU Calculation ---
        iou = tp / (tp + fp + fn + smooth)
        per_class_iou[cls_idx] = iou.item()

    metrics_dict = {
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_dice": per_class_dice,
        "per_class_iou": per_class_iou,
    }

    return metrics_dict, present_foreground_classes


# --- Utility function for calculating epoch averages ---
def calculate_epoch_avg_metrics(metric_sums, class_counts, foreground_classes):
    """Calculates average per-class and macro metrics for an epoch."""
    avg_per_class_precision = {}
    avg_per_class_recall = {}
    avg_per_class_dice = {}
    avg_per_class_iou = {}

    valid_precisions = []
    valid_recalls = []
    valid_dices = []
    valid_ious = []

    for cls_idx in foreground_classes:
        count = class_counts[cls_idx]
        if count > 0:
            avg_p = metric_sums["precision"][cls_idx] / count
            avg_r = metric_sums["recall"][cls_idx] / count
            avg_d = metric_sums["dice"][cls_idx] / count
            avg_i = metric_sums["iou"][cls_idx] / count

            avg_per_class_precision[cls_idx] = avg_p
            avg_per_class_recall[cls_idx] = avg_r
            avg_per_class_dice[cls_idx] = avg_d
            avg_per_class_iou[cls_idx] = avg_i

            # Collect valid metrics for macro calculation
            valid_precisions.append(avg_p)
            valid_recalls.append(avg_r)
            valid_dices.append(avg_d)
            valid_ious.append(avg_i)
        else:
            # If a class was never present, its average metric is 0 or undefined.
            # Assign 0.0 for reporting consistency. It won't contribute to macro average below.
            avg_per_class_precision[cls_idx] = 0.0
            avg_per_class_recall[cls_idx] = 0.0
            avg_per_class_dice[cls_idx] = 0.0
            avg_per_class_iou[cls_idx] = 0.0

    # Calculate Macro Averages based on classes that WERE present during the epoch
    macro_avg_precision = np.mean(valid_precisions) if valid_precisions else 0.0
    macro_avg_recall = np.mean(valid_recalls) if valid_recalls else 0.0
    macro_avg_dice = np.mean(valid_dices) if valid_dices else 0.0
    macro_avg_iou = np.mean(valid_ious) if valid_ious else 0.0

    return {
        "per_class_precision": avg_per_class_precision,
        "macro_avg_precision": macro_avg_precision,
        "per_class_recall": avg_per_class_recall,
        "macro_avg_recall": macro_avg_recall,
        "per_class_dice": avg_per_class_dice,
        "macro_avg_dice": macro_avg_dice,
        "per_class_iou": avg_per_class_iou,
        "macro_avg_iou": macro_avg_iou
    }

cls = 4   # mask_path = image_path.replace("patch_512", "mask_3cls")    NUM_CLASSES = 4
fold = 1

best_log_file_path = f"/home/gjs/ESD_2025/Segment/results/{cls}cls_fold_{fold}_best_epoch_summary_5.0.txt"

# --- Data Loading ---
train_dataset = SegmentationDataset(f"/home/gjs/ESD_2025/Segment/5fold_data/fold_{fold}_train.txt", transform=train_transform)
val_dataset = SegmentationDataset(f"/home/gjs/ESD_2025/Segment/5fold_data/fold_{fold}_val.txt", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4, pin_memory=True)

# --- Model, Loss, Optimizer ---
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
)

criterion_dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True, classes=FOREGROUND_CLASSES)
# criterion_focal = smp.losses.FocalLoss(mode='multiclass', gamma=2.0, normalized=False) # Use Focal Loss
# criterion_ce = nn.CrossEntropyLoss()
# criterion_ce = criterion_focal # Assign focal loss to the variable used later, or rename

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# --- Training Loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# model.load_state_dict(torch.load('/home/gjs/ESD_2025/Segment/ckpt/4cls_fold_1_best_macro_dice_new.pth', map_location=device))

num_epochs = 60
best_val_macro_dice = 0.0

print(f"Using device: {device}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Foreground classes for metrics: {FOREGROUND_CLASSES}")

for epoch in range(num_epochs):

    # --- Training Phase ---
    model.train()
    epoch_train_loss = 0.0
    # Accumulators: Use defaultdict for easier handling of sums and counts
    train_metric_sums = {
        "precision": defaultdict(float),
        "recall": defaultdict(float),
        "dice": defaultdict(float),
        "iou": defaultdict(float)
    }
    train_class_counts = defaultdict(int) # Count batches where each class was present
    train_batches = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=True)
    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad()
        outputs = model(images)

        loss_dice = criterion_dice(outputs, labels)
        # loss_ce = criterion_ce(outputs, labels)
        loss = loss_dice

        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        # Calculate batch metrics and identify present classes
        with torch.no_grad():
             batch_metrics_dict, present_classes = calculate_batch_metrics(outputs, labels, NUM_CLASSES)

        # Accumulate metrics ONLY for classes present in this batch's labels
        for cls_idx in present_classes:
            train_metric_sums["precision"][cls_idx] += batch_metrics_dict["per_class_precision"][cls_idx]
            train_metric_sums["recall"][cls_idx] += batch_metrics_dict["per_class_recall"][cls_idx]
            train_metric_sums["dice"][cls_idx] += batch_metrics_dict["per_class_dice"][cls_idx]
            train_metric_sums["iou"][cls_idx] += batch_metrics_dict["per_class_iou"][cls_idx]
            train_class_counts[cls_idx] += 1 # Increment count for this class

        train_batches += 1
        loop.set_postfix(loss=loss.item()) # Simplified postfix

    # Calculate average training metrics for the epoch
    avg_train_metrics = calculate_epoch_avg_metrics(train_metric_sums, train_class_counts, FOREGROUND_CLASSES)
    avg_train_loss = epoch_train_loss / train_batches

    # --- Validation Phase ---
    model.eval()
    epoch_val_loss = 0.0
    # Accumulators for validation
    val_metric_sums = {
        "precision": defaultdict(float),
        "recall": defaultdict(float),
        "dice": defaultdict(float),
        "iou": defaultdict(float)
    }
    val_class_counts = defaultdict(int)
    val_batches = 0

    with torch.no_grad():
        loop_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=True)
        for images, labels in loop_val:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            outputs = model(images)

            loss_dice = criterion_dice(outputs, labels)
            # loss_ce = criterion_ce(outputs, labels)
            loss = loss_dice

            epoch_val_loss += loss.item()

            # Calculate batch metrics and identify present classes
            batch_metrics_dict, present_classes = calculate_batch_metrics(outputs, labels, NUM_CLASSES)

            # Accumulate metrics ONLY for classes present in this batch's labels
            for cls_idx in present_classes:
                val_metric_sums["precision"][cls_idx] += batch_metrics_dict["per_class_precision"][cls_idx]
                val_metric_sums["recall"][cls_idx] += batch_metrics_dict["per_class_recall"][cls_idx]
                val_metric_sums["dice"][cls_idx] += batch_metrics_dict["per_class_dice"][cls_idx]
                val_metric_sums["iou"][cls_idx] += batch_metrics_dict["per_class_iou"][cls_idx]
                val_class_counts[cls_idx] += 1

            val_batches += 1
            loop_val.set_postfix(loss=loss.item()) # Simplified postfix

    # Calculate average validation metrics for the epoch
    avg_val_metrics = calculate_epoch_avg_metrics(val_metric_sums, val_class_counts, FOREGROUND_CLASSES)
    avg_val_loss = epoch_val_loss / val_batches

    scheduler.step(avg_val_loss)

    # --- Reporting ---
    print(f"\n--- Epoch {epoch + 1}/{num_epochs} Summary ---")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Train Macro Avg Precision: {avg_train_metrics['macro_avg_precision']:.4f}")
    print(f"  Train Macro Avg Recall:    {avg_train_metrics['macro_avg_recall']:.4f}")
    print(f"  Train Macro Avg Dice:      {avg_train_metrics['macro_avg_dice']:.4f}")
    print(f"  Train Macro Avg IoU:       {avg_train_metrics['macro_avg_iou']:.4f}")
    for cls in FOREGROUND_CLASSES:
        count = train_class_counts[cls]
        print(f"    Train P Class {cls}: {avg_train_metrics['per_class_precision'].get(cls, 0.0):.4f} (present in {count}/{train_batches} batches)")
        print(f"    Train R Class {cls}: {avg_train_metrics['per_class_recall'].get(cls, 0.0):.4f}")
        print(f"    Train D Class {cls}: {avg_train_metrics['per_class_dice'].get(cls, 0.0):.4f}")
        print(f"    Train I Class {cls}: {avg_train_metrics['per_class_iou'].get(cls, 0.0):.4f}")

    print("-" * 20)
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Val Macro Avg Precision: {avg_val_metrics['macro_avg_precision']:.4f}")
    print(f"  Val Macro Avg Recall:    {avg_val_metrics['macro_avg_recall']:.4f}")
    print(f"  Val Macro Avg Dice:      {avg_val_metrics['macro_avg_dice']:.4f}")
    print(f"  Val Macro Avg IoU:       {avg_val_metrics['macro_avg_iou']:.4f}")
    for cls in FOREGROUND_CLASSES:
        count = val_class_counts[cls]
        print(f"    Val P Class {cls}: {avg_val_metrics['per_class_precision'].get(cls, 0.0):.4f} (present in {count}/{val_batches} batches)")
        print(f"    Val R Class {cls}: {avg_val_metrics['per_class_recall'].get(cls, 0.0):.4f}")
        print(f"    Val D Class {cls}: {avg_val_metrics['per_class_dice'].get(cls, 0.0):.4f}")
        print(f"    Val I Class {cls}: {avg_val_metrics['per_class_iou'].get(cls, 0.0):.4f}")
    print("-" * 20)

    current_val_macro_dice = avg_val_metrics['macro_avg_dice']
    current_val_macro_iou = avg_val_metrics['macro_avg_iou'] # Get current IoU

    # --- Save Best Model (Based on Validation Macro Dice) ---
    if current_val_macro_dice > best_val_macro_dice:
        best_val_macro_dice = current_val_macro_dice
        best_epoch = epoch + 1
        # Save the model state dict
        model_save_path = f"/home/gjs/ESD_2025/Segment/ckpt/{cls}cls_fold_{fold}_best_macro_dice_5.0.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Epoch {best_epoch}: New best model saved to {model_save_path} with Macro Dice: {best_val_macro_dice:.4f}")

        # --- Write Summary to File for the Best Epoch ---
        try:
            with open(best_log_file_path, 'w') as f:
                f.write(f"--- Epoch {best_epoch}/{num_epochs} Summary (New Best Val Macro Dice) ---\n")
                f.write(f"Achieved Best Validation Macro Dice: {best_val_macro_dice:.4f}\n")
                f.write(f"Corresponding Validation Macro IoU: {current_val_macro_iou:.4f}\n") # Log corresponding IoU
                f.write("-" * 30 + "\n\n")

                f.write("--- Training Metrics ---\n")
                f.write(f"  Train Loss: {avg_train_loss:.4f}\n")
                f.write(f"  Train Macro Avg Precision: {avg_train_metrics['macro_avg_precision']:.4f}\n")
                f.write(f"  Train Macro Avg Recall:    {avg_train_metrics['macro_avg_recall']:.4f}\n")
                f.write(f"  Train Macro Avg Dice:      {avg_train_metrics['macro_avg_dice']:.4f}\n")
                f.write(f"  Train Macro Avg IoU:       {avg_train_metrics['macro_avg_iou']:.4f}\n")
                for c in FOREGROUND_CLASSES:
                    count = train_class_counts[c]
                    f.write(f"    Train P Class {c}: {avg_train_metrics['per_class_precision'].get(c, 0.0):.4f} (present in {count}/{train_batches} batches)\n")
                    f.write(f"    Train R Class {c}: {avg_train_metrics['per_class_recall'].get(c, 0.0):.4f}\n")
                    f.write(f"    Train D Class {c}: {avg_train_metrics['per_class_dice'].get(c, 0.0):.4f}\n")
                    f.write(f"    Train I Class {c}: {avg_train_metrics['per_class_iou'].get(c, 0.0):.4f}\n")

                f.write("\n" + "-" * 20 + "\n\n")

                f.write("--- Validation Metrics ---\n")
                f.write(f"  Val Loss: {avg_val_loss:.4f}\n")
                f.write(f"  Val Macro Avg Precision: {avg_val_metrics['macro_avg_precision']:.4f}\n")
                f.write(f"  Val Macro Avg Recall:    {avg_val_metrics['macro_avg_recall']:.4f}\n")
                f.write(f"  Val Macro Avg Dice:      {avg_val_metrics['macro_avg_dice']:.4f}\n")
                f.write(f"  Val Macro Avg IoU:       {avg_val_metrics['macro_avg_iou']:.4f}\n")
                for c in FOREGROUND_CLASSES:
                    count = val_class_counts[c]
                    f.write(f"    Val P Class {c}: {avg_val_metrics['per_class_precision'].get(c, 0.0):.4f} (present in {count}/{val_batches} batches)\n")
                    f.write(f"    Val R Class {c}: {avg_val_metrics['per_class_recall'].get(c, 0.0):.4f}\n")
                    f.write(f"    Val D Class {c}: {avg_val_metrics['per_class_dice'].get(c, 0.0):.4f}\n")
                    f.write(f"    Val I Class {c}: {avg_val_metrics['per_class_iou'].get(c, 0.0):.4f}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Best Validation Macro Dice achieved: {best_val_macro_dice:.4f} at epoch {best_epoch}")
            print(f"Best epoch summary saved to {best_log_file_path}")
        except IOError as e:
            print(f"Error writing best epoch summary to file: {e}")

    print("-" * 30)


print("Training finished!")
if best_val_macro_dice > 0.0:
     print(f"Best Validation Macro Dice achieved: {best_val_macro_dice:.4f} at epoch {best_epoch}")