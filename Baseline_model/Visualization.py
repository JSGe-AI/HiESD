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

input_size = 1024
hidden_size = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scaling_factor = 64    # <--- Each original coordinate is reduced by a factor of 64
block_size = 16        # <--- Each reduced coordinate represents a 16x16 block
# ****************************

model_name = "UNI"
target_labels = [0, 1,2,3,4]
num_classes = len(target_labels)


# ### Four Classes
color_palette = {
    4: (0, 0, 0),        # Background - Black
    0: (250, 0,250),   # Class 0 - Purple
    1: (250, 0, 0),      # Class 1 - Red
    2: (0, 250, 0),      # Class 2 - Green
    3: (0, 0, 250)       # Class 3 - Blue

}

# ### Three Classes
# color_palette = {
#     0: (0, 0, 0),        # Background - Black
#     1: (250, 0, 0),      # Class 1 (originally predicted as 0) - Red
#     2: (30, 144, 255),      # Class 2 (originally predicted as 1) - Green
#     3: (255, 215, 0),      # Class 3 (originally predicted as 2) - Blue
# }

# ### Two Classes
# color_palette = {
#     0: (0, 0, 0),        # Background - Black
#     1: (51, 51, 254),      # Class 1 (originally predicted as 0) - Red
#     2: (255, 51, 255),      # Class 2 (originally predicted as 1) - Green
# }


default_color = (255, 255, 255)

model_paths = [f"/home/gjs/ESD_2025/Experiment/ckpt/{model_name}/{num_classes}cls_best_fold_{i+1}.pth" for i in range(5)]
data_root_dir = f"/data_nas2/gjs/ESD_2025/classification/{model_name}_{num_classes}cls/5_fold/"
output_dir_base = f"./{model_name}_{num_classes}cls_predmaps_scaled{scaling_factor}_block{block_size}_v2" # New directory name
os.makedirs(output_dir_base, exist_ok=True)

print(f"Using device: {device}")
print(f"Number of model prediction classes: {num_classes}")
print(f"Target labels: {target_labels}")
print(f"Coordinate scaling factor: {scaling_factor}")
print(f"Final tile size: {block_size}x{block_size}")
print(f"Color mapping: {color_palette}")

# --------------- Evaluation and prediction map generation ---------------
for fold in range(5):
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

    # --- Create Fold Output Directory ---
    # fold_output_dir = os.path.join(output_dir_base, f"fold_{fold_num}")
    fold_output_dir = output_dir_base
    os.makedirs(fold_output_dir, exist_ok=True)

    # --- Process Each H5 File ---
    for h5_path in h5_files:
        h5_filename = os.path.basename(h5_path)
        print(f"  Processing file: {h5_filename}")

        try:
            # --- Data Loading and Checking ---
            with h5py.File(h5_path, 'r') as f:
                if not all(k in f for k in ['coords', 'features', 'labels']):
                    print(f"    Warning: Skipping {h5_filename} - Missing dataset.")
                    continue
                coords = f['coords'][()]
                features = f['features'][()]
                labels = f['labels'][()]
                n_patches = coords.shape[0]
                if n_patches == 0:
                    print(f"    Warning: Skipping {h5_filename} - No data.")
                    continue
                if not (features.shape[0] == n_patches and labels.shape[0] == n_patches):
                     print(f"    Warning: Skipping {h5_filename} - Dataset dimension mismatch.")
                     continue

            # --- Filter Data to be Predicted ---
            valid_mask = np.isin(labels, target_labels)
            filtered_coords = coords[valid_mask]
            filtered_features = features[valid_mask]
            num_valid_patches = filtered_features.shape[0]

            if num_valid_patches == 0:
                print(f"    Info: No valid label patches in {h5_filename}.")
                # Still need to calculate canvas size to generate possible empty or background images

            # --- Model Prediction ---
            predictions = np.array([], dtype=int)
            if num_valid_patches > 0:
                with torch.no_grad():
                    features_tensor = torch.tensor(filtered_features, dtype=torch.float32).to(device)
                    outputs = model(features_tensor)
                    pred_indices = torch.argmax(outputs, dim=1)
                    predictions = pred_indices.cpu().numpy()

            # --- Calculate Final Image Size ---
            # Use *all* original coordinates to determine the boundary
            if n_patches > 0: # Ensure coords is not empty
                # Scale all original coordinates down by scaling_factor
                scaled_all_coords = coords // scaling_factor
                # Find the maximum of the scaled coordinates
                max_scaled_x = np.max(scaled_all_coords[:, 0])
                max_scaled_y = np.max(scaled_all_coords[:, 1])
                # The width and height of the final image need to cover the edge of the bottom right block
                final_map_width = max_scaled_x + block_size
                final_map_height = max_scaled_y + block_size
            else: # If the file is empty, create a minimal canvas
                final_map_width = block_size
                final_map_height = block_size


            # --- Create Final Color Map ---
            # Create a blank final color map (RGB, uint8), with a default black background (0, 0, 0)
            final_color_map = np.zeros((final_map_height, final_map_width, 3), dtype=np.uint8)

            # --- Fill Final Color Map ---
            if num_valid_patches > 0:
                '''  3 classes/2 classes
                # Add 1 to the predicted label value (starting from 1)
                predicted_labels_plus_one = predictions + 1
                '''
                #5 classes
                predicted_labels_plus_one = predictions
                
                # Iterate through each *valid* patch
                for i in range(num_valid_patches):
                    # Get the original coordinates of the current patch
                    original_coord = filtered_coords[i]
                    # Calculate the top-left coordinate of the block on the final map
                    x_start = original_coord[0] // scaling_factor
                    y_start = original_coord[1] // scaling_factor
                    # Calculate the end coordinates of the block (exclusive)
                    x_end = x_start + block_size
                    y_end = y_start + block_size

                    # Get the color corresponding to the predicted label
                    label_value = predicted_labels_plus_one[i]
                    color = color_palette.get(label_value, default_color) # Use get to get the color, providing a default value

                    # Ensure coordinates are not out of bounds (although theoretically final_map should be large enough)
                    y_end = min(y_end, final_map_height)
                    x_end = min(x_end, final_map_width)

                    # Fill the color block using slices
                    final_color_map[y_start:y_end, x_start:x_end] = color

            # --- Save Final Color Prediction Map ---
            # output_filename = h5_filename.replace('.h5', f'_predmap_s{scaling_factor}_b{block_size}_v2.png')
            output_filename = h5_filename.replace('.h5', f'.png')
            output_png_path = os.path.join(fold_output_dir, output_filename)
            try:
                pil_img = Image.fromarray(final_color_map)
                pil_img.save(output_png_path)
                print(f"    Final color prediction map ({num_valid_patches}/{n_patches} valid patches) saved to: {output_png_path}")
            except Exception as e:
                print(f"    Error: Error saving final color image {output_png_path}: {e}")

        except Exception as e:
            print(f"  Error: Error processing file {h5_filename}: {e}")
            traceback.print_exc()

print("\n--- All v2 color prediction map generation complete ---")