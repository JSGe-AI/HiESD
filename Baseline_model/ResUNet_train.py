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
import argparse # Import argparse
import os # For path manipulation and creating directories

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global constants that might be derived from args later
# NUM_CLASSES will be set by args.num_classes
# FOREGROUND_CLASSES will be derived from NUM_CLASSES

class SegmentationDataset(Dataset):
    def __init__(self, image_txt_path, mask_replace_from="patch_512", mask_replace_to="mask_4cls", transform=None):
        self.image_paths = []
        self.mask_paths = []
        with open(image_txt_path, 'r') as f:
            for line in f:
                image_path = line.strip()
                self.image_paths.append(image_path)
                # Make mask path generation more flexible
                mask_path = image_path.replace(mask_replace_from, mask_replace_to)
                self.mask_paths.append(mask_path)
        self.transform = transform
        self.mask_replace_to = mask_replace_to # Store for potential debugging

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            mask = Image.open(self.mask_paths[idx]).convert("L") # Grayscale mask
        except FileNotFoundError as e:
            print(f"Error opening file: {e}")
            print(f"Attempted image: {self.image_paths[idx]}")
            print(f"Attempted mask: {self.mask_paths[idx]} (derived using replace_to='{self.mask_replace_to}')")
            raise e

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
def calculate_batch_metrics(preds, labels, num_classes, foreground_classes, smooth=1e-7):
    """
    Calculates per-class Precision, Recall, Dice score, IoU FOR A SINGLE BATCH.
    Crucially, it also identifies which foreground classes are present in the labels.

    Args:
        preds (torch.Tensor): Model output logits (B, C, H, W).
        labels (torch.Tensor): Ground truth labels (B, H, W).
        num_classes (int): Total number of classes (including background).
        foreground_classes (list): List of foreground class indices.
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
    present_foreground_classes = set(cls_idx for cls_idx in unique_labels.cpu().numpy() if cls_idx in foreground_classes)

    # Calculate metrics for all potential foreground classes
    for cls_idx in foreground_classes:
        pred_mask = (preds_argmax == cls_idx)
        true_mask = (labels == cls_idx)

        tp = (pred_mask & true_mask).sum().float()
        fp = (pred_mask & ~true_mask).sum().float()
        fn = (~pred_mask & true_mask).sum().float()

        precision = tp / (tp + fp + smooth)
        per_class_precision[cls_idx] = precision.item()

        recall = tp / (tp + fn + smooth)
        per_class_recall[cls_idx] = recall.item()

        dice = (2.0 * tp + smooth) / (tp + fp + tp + fn + smooth)
        per_class_dice[cls_idx] = dice.item()

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

            valid_precisions.append(avg_p)
            valid_recalls.append(avg_r)
            valid_dices.append(avg_d)
            valid_ious.append(avg_i)
        else:
            avg_per_class_precision[cls_idx] = 0.0
            avg_per_class_recall[cls_idx] = 0.0
            avg_per_class_dice[cls_idx] = 0.0
            avg_per_class_iou[cls_idx] = 0.0

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

def main(args):
    # --- Derived Global Variables ---
    NUM_CLASSES = args.num_classes
    FOREGROUND_CLASSES = list(range(1, NUM_CLASSES)) # Classes 1, ..., NUM_CLASSES-1

    # --- Output Paths ---
    os.makedirs(args.output_dir, exist_ok=True) # Ensure output directory exists
    
    # Construct filenames using the suffix
    filename_prefix = f"{NUM_CLASSES}cls_fold_{args.fold}"
    if args.suffix:
        filename_prefix += f"_{args.suffix}"

    best_log_file_path = os.path.join(args.output_dir, f"{filename_prefix}_best_epoch_summary.txt")
    model_save_path_template = os.path.join(args.output_dir, f"{filename_prefix}_best_macro_dice.pth")


    # --- Data Loading ---
    print(f"Loading train data from: {args.train_txt_path}")
    print(f"Loading val data from: {args.val_txt_path}")
    print(f"Masks derived by replacing '{args.mask_replace_from}' with '{args.mask_replace_to}' in image paths.")

    train_dataset = SegmentationDataset(
        args.train_txt_path,
        mask_replace_from=args.mask_replace_from,
        mask_replace_to=args.mask_replace_to,
        transform=train_transform
    )
    val_dataset = SegmentationDataset(
        args.val_txt_path,
        mask_replace_from=args.mask_replace_from,
        mask_replace_to=args.mask_replace_to,
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # --- Model, Loss, Optimizer ---
    model = smp.Unet(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=NUM_CLASSES,
    )

    criterion_dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True, classes=FOREGROUND_CLASSES)
    # criterion_focal = smp.losses.FocalLoss(mode='multiclass', gamma=2.0, normalized=False)
    # criterion_ce = nn.CrossEntropyLoss()
    # criterion = criterion_focal # Or combine losses as needed

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # --- Training Loop ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            print(f"Loading pre-trained model from: {args.load_model_path}")
            model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        else:
            print(f"Warning: Pre-trained model path not found: {args.load_model_path}. Starting from scratch.")


    best_val_macro_dice = 0.0
    best_epoch = 0

    print(f"Using device: {device}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Foreground classes for metrics: {FOREGROUND_CLASSES}")
    print(f"Training for {args.epochs} epochs.")
    print(f"Output directory: {args.output_dir}")

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0
        train_metric_sums = {
            "precision": defaultdict(float), "recall": defaultdict(float),
            "dice": defaultdict(float), "iou": defaultdict(float)
        }
        train_class_counts = defaultdict(int)
        train_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=True)
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion_dice(outputs, labels) # Using Dice loss, can be made configurable
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            with torch.no_grad():
                 batch_metrics_dict, present_classes = calculate_batch_metrics(outputs, labels, NUM_CLASSES, FOREGROUND_CLASSES)

            for cls_idx in present_classes:
                train_metric_sums["precision"][cls_idx] += batch_metrics_dict["per_class_precision"][cls_idx]
                train_metric_sums["recall"][cls_idx] += batch_metrics_dict["per_class_recall"][cls_idx]
                train_metric_sums["dice"][cls_idx] += batch_metrics_dict["per_class_dice"][cls_idx]
                train_metric_sums["iou"][cls_idx] += batch_metrics_dict["per_class_iou"][cls_idx]
                train_class_counts[cls_idx] += 1
            train_batches += 1
            loop.set_postfix(loss=loss.item())

        avg_train_metrics = calculate_epoch_avg_metrics(train_metric_sums, train_class_counts, FOREGROUND_CLASSES)
        avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0

        # --- Validation Phase ---
        model.eval()
        epoch_val_loss = 0.0
        val_metric_sums = {
            "precision": defaultdict(float), "recall": defaultdict(float),
            "dice": defaultdict(float), "iou": defaultdict(float)
        }
        val_class_counts = defaultdict(int)
        val_batches = 0

        with torch.no_grad():
            loop_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", leave=True)
            for images, labels in loop_val:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()
                outputs = model(images)
                loss = criterion_dice(outputs, labels)
                epoch_val_loss += loss.item()

                batch_metrics_dict, present_classes = calculate_batch_metrics(outputs, labels, NUM_CLASSES, FOREGROUND_CLASSES)
                for cls_idx in present_classes:
                    val_metric_sums["precision"][cls_idx] += batch_metrics_dict["per_class_precision"][cls_idx]
                    val_metric_sums["recall"][cls_idx] += batch_metrics_dict["per_class_recall"][cls_idx]
                    val_metric_sums["dice"][cls_idx] += batch_metrics_dict["per_class_dice"][cls_idx]
                    val_metric_sums["iou"][cls_idx] += batch_metrics_dict["per_class_iou"][cls_idx]
                    val_class_counts[cls_idx] += 1
                val_batches += 1
                loop_val.set_postfix(loss=loss.item())

        avg_val_metrics = calculate_epoch_avg_metrics(val_metric_sums, val_class_counts, FOREGROUND_CLASSES)
        avg_val_loss = epoch_val_loss / val_batches if val_batches > 0 else 0
        scheduler.step(avg_val_loss)

        # --- Reporting ---
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} Summary ---")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train Macro Avg Precision: {avg_train_metrics['macro_avg_precision']:.4f}")
        print(f"  Train Macro Avg Recall:    {avg_train_metrics['macro_avg_recall']:.4f}")
        print(f"  Train Macro Avg Dice:      {avg_train_metrics['macro_avg_dice']:.4f}")
        print(f"  Train Macro Avg IoU:       {avg_train_metrics['macro_avg_iou']:.4f}")
        for cls_label in FOREGROUND_CLASSES:
            count = train_class_counts[cls_label]
            # Use .get(cls_label, 0.0) for robustness if a class somehow isn't in the dict
            print(f"    Train P Class {cls_label}: {avg_train_metrics['per_class_precision'].get(cls_label, 0.0):.4f} (present in {count}/{train_batches} batches)")
            print(f"    Train R Class {cls_label}: {avg_train_metrics['per_class_recall'].get(cls_label, 0.0):.4f}")
            print(f"    Train D Class {cls_label}: {avg_train_metrics['per_class_dice'].get(cls_label, 0.0):.4f}")
            print(f"    Train I Class {cls_label}: {avg_train_metrics['per_class_iou'].get(cls_label, 0.0):.4f}")

        print("-" * 20)
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Macro Avg Precision: {avg_val_metrics['macro_avg_precision']:.4f}")
        print(f"  Val Macro Avg Recall:    {avg_val_metrics['macro_avg_recall']:.4f}")
        print(f"  Val Macro Avg Dice:      {avg_val_metrics['macro_avg_dice']:.4f}")
        print(f"  Val Macro Avg IoU:       {avg_val_metrics['macro_avg_iou']:.4f}")
        for cls_label in FOREGROUND_CLASSES:
            count = val_class_counts[cls_label]
            print(f"    Val P Class {cls_label}: {avg_val_metrics['per_class_precision'].get(cls_label, 0.0):.4f} (present in {count}/{val_batches} batches)")
            print(f"    Val R Class {cls_label}: {avg_val_metrics['per_class_recall'].get(cls_label, 0.0):.4f}")
            print(f"    Val D Class {cls_label}: {avg_val_metrics['per_class_dice'].get(cls_label, 0.0):.4f}")
            print(f"    Val I Class {cls_label}: {avg_val_metrics['per_class_iou'].get(cls_label, 0.0):.4f}")
        print("-" * 20)

        current_val_macro_dice = avg_val_metrics['macro_avg_dice']
        current_val_macro_iou = avg_val_metrics['macro_avg_iou']

        if current_val_macro_dice > best_val_macro_dice:
            best_val_macro_dice = current_val_macro_dice
            best_epoch = epoch + 1
            
            # Use the template for saving the model
            final_model_save_path = model_save_path_template
            torch.save(model.state_dict(), final_model_save_path)
            print(f"Epoch {best_epoch}: New best model saved to {final_model_save_path} with Macro Dice: {best_val_macro_dice:.4f}")

            try:
                with open(best_log_file_path, 'w') as f:
                    f.write(f"--- Epoch {best_epoch}/{args.epochs} Summary (New Best Val Macro Dice) ---\n")
                    f.write(f"Achieved Best Validation Macro Dice: {best_val_macro_dice:.4f}\n")
                    f.write(f"Corresponding Validation Macro IoU: {current_val_macro_iou:.4f}\n")
                    f.write(f"Model saved to: {final_model_save_path}\n")
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
                print(f"Best epoch summary saved to {best_log_file_path}")
            except IOError as e:
                print(f"Error writing best epoch summary to file: {e}")
        print("-" * 30)

    print("Training finished!")
    if best_val_macro_dice > 0.0:
         print(f"Best Validation Macro Dice achieved: {best_val_macro_dice:.4f} at epoch {best_epoch}")
         print(f"Best model and summary log can be found in: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Segmentation Model")

    # Path arguments
    parser.add_argument("--train_txt_path", type=str, required=True,
                        help="Path to the .txt file listing training images.")
    parser.add_argument("--val_txt_path", type=str, required=True,
                        help="Path to the .txt file listing validation images.")
    parser.add_argument("--output_dir", type=str, default="./results_segmentation",
                        help="Directory to save checkpoints and logs.")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="Optional path to a .pth model file to load for fine-tuning or resuming.")

    # Dataset related arguments
    parser.add_argument("--mask_replace_from", type=str, default="image",
                        help="Substring in image path to replace for finding mask path.")
    parser.add_argument("--mask_replace_to", type=str, default="mask", # Example, align with num_classes
                        help="Substring to replace with for finding mask path (e.g., 'mask_4cls' for 5 classes).")

    # Model and Training arguments
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Total number of classes (background + foreground objects).")
    parser.add_argument("--fold", type=int, default=1,
                        help="Fold number, used for output file naming.")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training and validation batch size.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate.")
    parser.add_argument("--encoder_name", type=str, default="resnet50",
                        help="Name of the encoder for smp.Unet (e.g., 'resnet50', 'efficientnet-b0').")
    parser.add_argument("--encoder_weights", type=str, default="imagenet",
                        help="Pretrained weights for the encoder (e.g., 'imagenet', None).")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix to add to output filenames (e.g., 'v1', 'test_run').")


    args = parser.parse_args()

    # You might want to add a sanity check for mask_replace_to based on num_classes if it follows a strict pattern
    # For example:
    # expected_mask_tag = f"mask_{args.num_classes-1}cls"
    # if args.mask_replace_to != expected_mask_tag:
    #     print(f"Warning: --mask_replace_to is '{args.mask_replace_to}', "
    #           f"but based on --num_classes={args.num_classes}, "
    #           f"you might expect '{expected_mask_tag}'. Ensure this is correct.")

    main(args)