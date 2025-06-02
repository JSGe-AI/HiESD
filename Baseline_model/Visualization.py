# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import h5py
from PIL import Image
import glob
import traceback
import argparse # Import argparse

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def get_color_palette(num_classes_arg):
    """
    Returns the color palette based on the number of classes.
    Also returns whether to add 1 to predictions for colormap indexing.
    """
    add_one_to_predictions = False
    if num_classes_arg == 5:
        # This palette maps predictions 0,1,2,3,4 directly.
        # Class 4 is treated as background (black)
        # Predictions are used directly (0 maps to purple, 1 to red, ..., 4 to black)
        palette = {
            4: (0, 0, 0),        # Class 4 (e.g., Background) - Black
            0: (250, 0, 250),    # Class 0 - Purple
            1: (250, 0, 0),      # Class 1 - Red
            2: (0, 250, 0),      # Class 2 - Green
            3: (0, 0, 250)       # Class 3 - Blue
        }
        add_one_to_predictions = False # Predictions are 0,1,2,3,4
    elif num_classes_arg == 3: # For example, 2 foreground classes + 1 background concept
        # Model predicts 0, 1, 2.
        # Predictions are shifted: pred 0 -> key 1, pred 1 -> key 2, pred 2 -> key 3
        # Key 0 is explicit background
        palette = {
            0: (0, 0, 0),        # Background - Black
            1: (250, 0, 0),      # Class 1 (model predicted 0) - Red
            2: (30, 144, 255),   # Class 2 (model predicted 1) - Dodger Blue
            3: (255, 215, 0),    # Class 3 (model predicted 2) - Gold
        }
        add_one_to_predictions = True # Predictions are 0,1,2
    elif num_classes_arg == 2: # For example, 1 foreground class + 1 background concept
        # Model predicts 0, 1.
        # Predictions are shifted: pred 0 -> key 1, pred 1 -> key 2
        # Key 0 is explicit background
        palette = {
            0: (0, 0, 0),        # Background - Black
            1: (51, 51, 254),    # Class 1 (model predicted 0) - Blueish
            2: (255, 51, 255),   # Class 2 (model predicted 1) - Pinkish
        }
        add_one_to_predictions = True # Predictions are 0,1
    else:
        print(f"Warning: No specific color palette defined for {num_classes_arg} classes. Generating a random one.")
        palette = {0: (0, 0, 0)} # Background
        # Assuming model predicts 0 to num_classes_arg-1
        # We'll map prediction `i` to palette key `i+1` (if add_one_to_predictions is True)
        # or prediction `i` to palette key `i` (if add_one_to_predictions is False)
        # For consistency with 2/3 class cases, let's assume if not 5, we do +1 mapping.
        # If num_classes is the actual number of output neurons.
        if num_classes_arg != 5: # Default behavior for other class numbers
             add_one_to_predictions = True
             for i in range(num_classes_arg):
                palette[i+1] = (np.random.randint(50, 256), np.random.randint(50, 256), np.random.randint(50, 256))
        else: # For num_classes == 5, handled above, but as a fallback for other cases with direct mapping
            add_one_to_predictions = False
            for i in range(num_classes_arg):
                 palette[i] = (np.random.randint(50, 256), np.random.randint(50, 256), np.random.randint(50, 256))
                 if i == num_classes_arg -1 : # Make last class black if it's supposed to be background
                     palette[i] = (0,0,0)


    default_color = (255, 255, 255) # White for unmapped labels
    return palette, default_color, add_one_to_predictions

def main(args):
    # --- Setup from args ---
    input_size = args.input_size
    hidden_size = args.hidden_size
    scaling_factor = args.scaling_factor
    block_size = args.block_size
    model_name = args.model_name
    num_classes = args.num_classes
    num_folds = args.num_folds

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        if args.device == 'cuda':
            print("Warning: CUDA requested but not available, using CPU.")
        if args.device == 'mps':
            print("Warning: MPS requested but not available or not built, using CPU.")


    # Define target labels based on num_classes (assuming labels are 0 to num_classes-1)
    target_labels = list(range(num_classes))

    color_palette, default_color, add_one_to_pred_for_colormap = get_color_palette(num_classes)

    # Construct paths
    model_paths = [f"{args.ckpt_base_dir}/{model_name}/{num_classes}cls_best_fold_{i+1}.pth" for i in range(num_folds)]
    data_root_dir = f"{args.data_root_base_dir}/{model_name}_{num_classes}cls/{num_folds}_fold/"
    # Use args.output_dir directly if provided, otherwise construct it
    if args.output_dir:
        output_dir_final = args.output_dir
    else:
        output_dir_final = f"./{model_name}_{num_classes}cls_predmaps_scaled{scaling_factor}_block{block_size}_v_args"
    os.makedirs(output_dir_final, exist_ok=True)

    print(f"--- Configuration ---")
    print(f"Using device: {device}")
    print(f"Model Name: {model_name}")
    print(f"Number of prediction classes: {num_classes}")
    print(f"Target labels for filtering H5 data: {target_labels}")
    print(f"MLP Input Size: {input_size}, Hidden Size: {hidden_size}")
    print(f"Coordinate scaling factor: {scaling_factor}")
    print(f"Final tile/block size: {block_size}x{block_size}")
    print(f"Number of Folds: {num_folds}")
    print(f"Checkpoint Base Directory: {args.ckpt_base_dir}")
    print(f"Data Root Base Directory: {args.data_root_base_dir}")
    print(f"Output Directory: {output_dir_final}")
    print(f"Color mapping: {color_palette}")
    print(f"Add 1 to predictions for colormap: {add_one_to_pred_for_colormap}")
    print(f"--------------------")


    # --------------- Evaluation and prediction map generation ---------------
    for fold in range(num_folds):
        fold_num = fold + 1
        print(f"\n--- Processing Fold {fold_num} ---")

        # --- Load Model ---
        model_path = model_paths[fold]
        if not os.path.exists(model_path):
            print(f"Warning: Fold {fold_num} model path not found: {model_path}")
            continue
        model = MLP(input_size, hidden_size, num_classes).to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error: Error loading Fold {fold_num} model state: {e}")
            continue
        model.eval()
        print(f"Loaded model: {model_path}")

        # --- Find H5 Files ---
        h5_dir = os.path.join(data_root_dir, f"fold_{fold_num}", "val")
        if not os.path.isdir(h5_dir):
            print(f"Warning: Fold {fold_num} H5 directory not found: {h5_dir}")
            continue
        h5_files = glob.glob(os.path.join(h5_dir, '*.h5'))
        if not h5_files:
            print(f"Warning: No H5 files found in {h5_dir}")
            continue
        print(f"Found {len(h5_files)} H5 files")

        # Output for this fold will go into the main output_dir_final
        # os.makedirs(output_dir_final, exist_ok=True) # Already created above

        # --- Process Each H5 File ---
        for h5_path in h5_files:
            h5_filename = os.path.basename(h5_path)
            print(f"  Processing file: {h5_filename}")

            try:
                # --- Data Loading and Checking ---
                with h5py.File(h5_path, 'r') as f:
                    if not all(k in f for k in ['coords', 'features', 'labels']):
                        print(f"    Warning: Skipping {h5_filename} - Missing dataset (coords, features, or labels).")
                        continue
                    coords = f['coords'][()]
                    features = f['features'][()]
                    labels = f['labels'][()] # These are ground truth labels in the H5 file
                    n_patches = coords.shape[0]
                    if n_patches == 0:
                        print(f"    Warning: Skipping {h5_filename} - No data.")
                        continue
                    if not (features.shape[0] == n_patches and labels.shape[0] == n_patches):
                         print(f"    Warning: Skipping {h5_filename} - Dataset dimension mismatch.")
                         continue

                # --- Filter Data to be Predicted based on target_labels ---
                # target_labels are [0, 1, ..., num_classes-1]
                # We only process patches whose ground truth label is one of these.
                valid_mask = np.isin(labels, target_labels)
                filtered_coords = coords[valid_mask]
                filtered_features = features[valid_mask]
                num_valid_patches = filtered_features.shape[0]

                if num_valid_patches == 0:
                    print(f"    Info: No patches with target labels {target_labels} in {h5_filename}.")
                    # Still need to calculate canvas size to generate possible empty or background images

                # --- Model Prediction ---
                model_predictions = np.array([], dtype=int) # Raw model predictions (0 to num_classes-1)
                if num_valid_patches > 0:
                    with torch.no_grad():
                        features_tensor = torch.tensor(filtered_features, dtype=torch.float32).to(device)
                        outputs = model(features_tensor)
                        pred_indices = torch.argmax(outputs, dim=1)
                        model_predictions = pred_indices.cpu().numpy()

                # --- Calculate Final Image Size ---
                if n_patches > 0:
                    scaled_all_coords = coords // scaling_factor
                    max_scaled_x = np.max(scaled_all_coords[:, 0])
                    max_scaled_y = np.max(scaled_all_coords[:, 1])
                    final_map_width = max_scaled_x + block_size
                    final_map_height = max_scaled_y + block_size
                else:
                    final_map_width = block_size
                    final_map_height = block_size

                # --- Create Final Color Map ---
                final_color_map = np.zeros((final_map_height, final_map_width, 3), dtype=np.uint8) # Default black

                # --- Fill Final Color Map ---
                if num_valid_patches > 0:
                    if add_one_to_pred_for_colormap:
                        colormap_keys = model_predictions + 1
                    else:
                        colormap_keys = model_predictions

                    for i in range(num_valid_patches):
                        original_coord = filtered_coords[i]
                        x_start = original_coord[0] // scaling_factor
                        y_start = original_coord[1] // scaling_factor
                        x_end = x_start + block_size
                        y_end = y_start + block_size

                        label_value_for_colormap = colormap_keys[i]
                        color = color_palette.get(label_value_for_colormap, default_color)

                        y_end = min(y_end, final_map_height)
                        x_end = min(x_end, final_map_width)

                        final_color_map[y_start:y_end, x_start:x_end] = color

                # --- Save Final Color Prediction Map ---
                output_filename = h5_filename.replace('.h5', f'_fold{fold_num}.png') # Add fold info to filename
                output_png_path = os.path.join(output_dir_final, output_filename)
                try:
                    pil_img = Image.fromarray(final_color_map)
                    pil_img.save(output_png_path)
                    print(f"    Final color prediction map ({num_valid_patches}/{n_patches} valid patches) saved to: {output_png_path}")
                except Exception as e:
                    print(f"    Error: Error saving final color image {output_png_path}: {e}")

            except Exception as e:
                print(f"  Error: Error processing file {h5_filename}: {e}")
                traceback.print_exc()

    print(f"\n--- All prediction map generation complete. Output in {output_dir_final} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prediction maps from H5 features using a trained MLP model.")

    # Model architecture
    parser.add_argument('--input_size', type=int, default=1024, help='Input size for the MLP.')
    parser.add_argument('--hidden_size', type=int, default=2048, help='Hidden size for the MLP.')
    parser.add_argument('--num_classes', type=int, default=5, choices=[2, 3, 5], help='Number of output classes for the model. Affects color palette.')

    # Processing parameters
    parser.add_argument('--scaling_factor', type=int, default=64, help='Factor by which original coordinates are reduced.')
    parser.add_argument('--block_size', type=int, default=16, help='Size of the block (e.g., 16x16) each reduced coordinate represents in the final map.')

    # Naming and Paths
    parser.add_argument('--model_name', type=str, default="UNI", help='Name of the model (used in path construction).')
    parser.add_argument('--ckpt_base_dir', type=str, default="/home/gjs/ESD_2025/Experiment/ckpt/", help='Base directory for model checkpoints.')
    parser.add_argument('--data_root_base_dir', type=str, default="/data_nas2/gjs/ESD_2025/classification/", help='Base directory for H5 data.')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for prediction maps. If None, a default is constructed.')

    # Experiment setup
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds to process.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'], help='Device to use for inference (cuda, cpu, mps).')

    args = parser.parse_args()
    main(args)