import os
import argparse #  argparse
from openslide import OpenSlide
from PIL import Image
import numpy as np
import shutil


def save_svs_patches(svs_path, output_dir, patch_size=512):
    slide = OpenSlide(svs_path)
    wsi_width, wsi_height = slide.dimensions
    num_rows = wsi_height // patch_size
    num_cols = wsi_width // patch_size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for row in range(num_rows):
        for col in range(num_cols):
            x = col * patch_size
            y = row * patch_size
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch_filename = f"{x}_{y}_{patch_size}.png" 
            patch.save(os.path.join(output_dir, patch_filename))
    slide.close()
    print(f"Saved patches from {svs_path} to {output_dir}")

def delete_backgrounds(path, white_ratio_threshold=0.75): #  white_ratio_threshold 
    image_files = [file for file in os.listdir(path) if file.endswith('.jpg') or file.endswith('.png')]
    deleted_count = 0
    for image_file in image_files:
        image_path = os.path.join(path, image_file)
        try:
            image = Image.open(image_path).convert('L')
            image = image.point(lambda x: 255 if x > 128 else 0)
            white_pixels = sum(pixel == 255 for pixel in image.getdata())
            total_pixels = image.width * image.height
            if total_pixels == 0: continue # Avoid division by zero for empty/corrupt images

            white_ratio = white_pixels / total_pixels
            if white_ratio > white_ratio_threshold:
                os.remove(image_path)
                # print(f"Deleted {image_file} due to high white ratio ({white_ratio:.2f}).")
                deleted_count += 1
        except Exception as e:
            print(f"Error processing {image_path} in delete_backgrounds: {e}")
    if deleted_count > 0:
        print(f"Deleted {deleted_count} background patches from {path}")

def process_images(directory):

    print(f"Processing images in {directory} (color mapping) - Note: This function was not called in the original main loop.")
    image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.png') or file.endswith('.jpg')]
    for image_file in image_files:
        try:
            image = Image.open(image_file).convert('RGB')
            pixels = image.load()
            for i in range(image.width):
                for j in range(image.height):
                    r, g, b = pixels[i, j]
                    if (r, g, b) == (1, 1, 1): pixels[i, j] = (255, 0, 0)
                    elif (r, g, b) == (2, 2, 2): pixels[i, j] = (0, 255, 0)
                    elif (r, g, b) == (3, 3, 3): pixels[i, j] = (0, 0, 255)
                    elif (r, g, b) == (4, 4, 4): pixels[i, j] = (255, 0, 255)
                    elif (r, g, b) == (5, 5, 5): pixels[i, j] = (255, 192, 203)
                    elif (r, g, b) == (6, 6, 6): pixels[i, j] = (255, 255, 0)
                    elif (r, g, b) == (7, 7, 7): pixels[i, j] = (218, 112, 214)
                    elif (r, g, b) == (8, 8, 8): pixels[i, j] = (25, 25, 112)
            image.save(image_file)
        except Exception as e:
            print(f"Error processing {image_file} in process_images: {e}")


def bag_involved(patch_directory, comp_image_path):
    categories = {
        '0': (1, 1, 1), '1': (2, 2, 2), '2': (3, 3, 3), '3': (4, 4, 4),
        '4': (5, 5, 5), '5': (6, 6, 6), '6': (7, 7, 7),
    }
    category_bags = {}
    for i in range(7): # Corresponds to categories '0' through '6'
        folder_path = os.path.join(patch_directory, f"c{i+1}") # Folders c1 to c7
        os.makedirs(folder_path, exist_ok=True)
        category_bags[f'{i}'] = folder_path # Map category '0' to c1, '1' to c2 etc.

    moved_count = 0
    if not os.path.exists(comp_image_path):
        print(f"Warning: Component image path {comp_image_path} does not exist. Skipping bag_involved for {patch_directory}.")
        return

    target_image = Image.open(comp_image_path).convert('RGB')
    target_width, target_height = target_image.size

    for filename in os.listdir(patch_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            patch_image_path = os.path.join(patch_directory, filename)
            name_parts = filename.split('_')
            if len(name_parts) >= 2: # Ensure at least x and y are present
                try:
                    x = int(name_parts[0])
                    y = int(name_parts[1]) # .split('.')[0] is handled by int() if "Y.png"

                    target_x = x // 64
                    target_y = y // 64

                    if target_x < target_width and target_y < target_height:
                        start_x = target_x
                        start_y = target_y
                        end_x = min(start_x + 16, target_width) # When modifying patch_size, please modify the step size. (For example: 1024/64=16)  
                        end_y = min(start_y + 16, target_height) # Ensure crop doesn't go out of bounds

                        if start_x >= end_x or start_y >= end_y: continue # Skip if ROI is invalid

                        roi_image = target_image.crop((start_x, start_y, end_x, end_y))
                        roi_array = np.array(roi_image)
                        if roi_array.size == 0: continue # Skip if ROI is empty

                        color_counts = {category: 0 for category in categories}
                        for category_key, color_val in categories.items():
                            # Count occurrences of each color tuple
                            mask = np.all(roi_array == color_val, axis=-1)
                            color_counts[category_key] = np.sum(mask)
                        
                        # Ensure there's at least one color found, otherwise max on empty will fail
                        if not any(color_counts.values()):
                            # print(f"No defined category colors found in ROI for {filename}. Skipping.")
                            continue

                        most_common_category_key = max(color_counts, key=color_counts.get)
                        
                        destination_bag_path = category_bags[most_common_category_key]
                        new_image_path = os.path.join(destination_bag_path, filename)
                        shutil.move(patch_image_path, new_image_path)
                        moved_count +=1
                except ValueError:
                    # print(f"Could not parse coordinates from filename: {filename}. Skipping.")
                    continue # If filename isn't in expected x_y_size.ext format
                except Exception as e:
                    print(f"Error processing {filename} in bag_involved: {e}")
    target_image.close()
    if moved_count > 0:
        print(f"Moved {moved_count} patches into category bags within {patch_directory} based on {comp_image_path}")

def label_projection(bag_dic, mask_dic_path): # Renamed mask_dic to mask_dic_path for clarity
    categories = {
        '1': (139, 0, 0), '2': (255, 0, 255), '3': (128, 0, 128), '4': (75, 0, 130),
        '5': (138, 43, 226), '6': (0, 0, 255), '7': (65, 105, 225), '8': (70, 130, 180),
        '9': (0, 255, 255), '10': (0, 255, 0), '11': (0, 128, 0), '12': (255, 255, 0),
        '13': (255, 215, 0), '14': (255, 165, 0), '15': (255, 0, 0),
    }
    # Folder "0" for high black ratio, "1" through "15" for categories
    for i in range(16): # Folders 0 to 15
        folder_path = os.path.join(bag_dic, f"{i}")
        os.makedirs(folder_path, exist_ok=True)

    destination_paths = {str(i): os.path.join(bag_dic, str(i)) for i in range(16)}
    
    moved_count = 0
    if not os.path.exists(mask_dic_path):
        print(f"Warning: Mask dictionary path {mask_dic_path} does not exist. Skipping label_projection for {bag_dic}.")
        return

    target_image = Image.open(mask_dic_path).convert('RGB')
    target_width, target_height = target_image.size

    # Iterate over files directly in bag_dic, as they are moved there before this function is called on subfolders
    # This assumes files are in bag_dic (e.g. c1, c2...) and need to be moved to subfolders (0,1,2...) within bag_dic
    patch_files_to_process = [f for f in os.listdir(bag_dic) if (f.endswith('.jpg') or f.endswith('.png')) and os.path.isfile(os.path.join(bag_dic, f))]

    for patch_name in patch_files_to_process:
        patch_image_path = os.path.join(bag_dic, patch_name)
        name_parts = patch_name.split('_')
        if len(name_parts) >= 2:
            try:
                x = int(name_parts[0])
                y = int(name_parts[1])

                target_x = x // 64
                target_y = y // 64

                if target_x < target_width and target_y < target_height:
                    start_x = target_x
                    start_y = target_y
                    end_x = min(start_x + 16, target_width) # When modifying patch_size, please modify the step size. (For example: 1024/64=16) 
                    end_y = min(start_y + 16, target_height)

                    if start_x >= end_x or start_y >= end_y: continue

                    roi_image = target_image.crop((start_x, start_y, end_x, end_y))
                    roi_array = np.array(roi_image)
                    if roi_array.size == 0: continue
                    
                    roi_width, roi_height = roi_image.size
                    total_pixels_roi = roi_width * roi_height
                    if total_pixels_roi == 0: continue

                    black_pixel_count = np.sum(np.all(roi_array == [0, 0, 0], axis=2))
                    black_pixel_ratio = black_pixel_count / total_pixels_roi

                    if black_pixel_ratio > 0.9:
                        des_path = destination_paths["0"]
                    else:
                        color_counts = {category_key: 0 for category_key in categories}
                        for category_key, color_val in categories.items():
                            mask = np.all(roi_array == color_val, axis=-1)
                            color_counts[category_key] = np.sum(mask)
                        
                        if not any(color_counts.values()):
                            # print(f"No defined category colors found in ROI for {patch_name} (label_projection). Defaulting to '0'.")
                            des_path = destination_paths["0"] # Or handle as error/skip
                        else:
                            most_common_category_key = max(color_counts, key=color_counts.get)
                            des_path = destination_paths[most_common_category_key]
                    
                    shutil.move(patch_image_path, os.path.join(des_path, patch_name))
                    moved_count += 1
            except ValueError:
                # print(f"Could not parse coordinates from filename: {patch_name} in label_projection. Skipping.")
                continue
            except Exception as e:
                print(f"Error processing {patch_name} in label_projection for bag {bag_dic}: {e}")
    target_image.close()
    if moved_count > 0:
        print(f"Projected labels for {moved_count} patches in {bag_dic} using {mask_dic_path}")

def delete_empty_folders(folder_path_to_clean):
    # First pass: delete empty sub-sub-folders (like c1/0, c1/1 if empty)
    for root, dirs, files in os.walk(folder_path_to_clean, topdown=False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            if not os.listdir(dir_path): # Check if directory is empty
                try:
                    os.rmdir(dir_path)
                    print(f"Deleted empty sub-folder: {dir_path}")
                except OSError as e:
                    print(f"Error deleting {dir_path}: {e}")
    
    # Second pass: delete empty main category folders (like c1, c2 if they became empty)
    # This is implicitly handled if folder_path_to_clean is the parent of c1, c2, etc.
    # And if c1, c2 are themselves passed to this function in a loop.
    # The original code called delete_folders(final_data_path), which would clean up
    # empty svs_file folders and then potentially empty cX folders within them.
    # The current structure of os.walk(topdown=False) should handle this well.
    # If 'folder_path_to_clean' itself becomes empty, it's not deleted by this function.
    # This is usually desired to not delete the top-level output directory.
    print(f"Completed empty folder cleanup for subdirectories within: {folder_path_to_clean}")


def main():
    parser = argparse.ArgumentParser(description="Process SVS files, extract patches, and categorize them.")
    parser.add_argument("--svs_dir", type=str, required=True, help="Directory containing SVS patient/case subfolders.")
    parser.add_argument("--mask_components_dir", type=str, required=True, help="A directory containing all the svs files corresponding to the complete connected components (e.g., 'ESD_components_104').")
    parser.add_argument("--summary_dir", type=str, required=True, help="Directory for summary mask images (e.g., 'ESD_summary').")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save processed patches (e.g., 'final_data').")
    parser.add_argument("--patch_size", type=int, default=1024, help="Size of the patches to extract (default: 1024).")
    parser.add_argument("--white_ratio_threshold", type=float, default=0.75, help="Threshold for deleting background patches (default: 0.75).")
    
    args = parser.parse_args()

    # Create base output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created base output directory: {args.output_dir}")

    # Loop through patient/case folders in svs_dir
    for svs_case_folder_name in os.listdir(args.svs_dir):
        svs_case_folder_path = os.path.join(args.svs_dir, svs_case_folder_name)
        if not os.path.isdir(svs_case_folder_path):
            continue

        # This is the directory where patches for the current case will be stored temporarily
        # e.g., output_dir/TCGA-01-0001
        case_patch_output_dir = os.path.join(args.output_dir, svs_case_folder_name)
        
        if os.path.exists(case_patch_output_dir):
            print(f"Output directory {case_patch_output_dir} already exists. Skipping case {svs_case_folder_name}.")
            # If you want to re-process, you might want to shutil.rmtree(case_patch_output_dir) here
            # and then os.makedirs(case_patch_output_dir)
            continue
        os.makedirs(case_patch_output_dir, exist_ok=True)
        print(f"Processing case: {svs_case_folder_name}, outputting to {case_patch_output_dir}")

        # Loop through actual .svs files within the case folder
        for svs_filename in os.listdir(svs_case_folder_path):
            if not svs_filename.lower().endswith(('.svs', '.tif', '.tiff')): # Add other WSI formats if needed
                continue
            
            svs_file_full_path = os.path.join(svs_case_folder_path, svs_filename)
            print(f"  Processing SVS file: {svs_file_full_path}")

            # Patches from one SVS file go into the case_patch_output_dir directly
            # The original code saved all SVS from one svs_file_path (case) into one patch_directory
            save_svs_patches(svs_file_full_path, case_patch_output_dir, patch_size=args.patch_size)
            delete_backgrounds(case_patch_output_dir, white_ratio_threshold=args.white_ratio_threshold)
            
            # Component and summary mask paths
            # Assuming svs_case_folder_name is like 'TCGA-01-0001'
            comp_file_name = svs_case_folder_name + '.png' # Original logic
            comp_image_path = os.path.join(args.mask_components_dir, comp_file_name)
            
            mask_image_name = svs_case_folder_name + '.png' # Original logic
            mask_image_path = os.path.join(args.summary_dir, mask_image_name)

            # Bagging based on component mask
            print(f"  Performing bag involvement for: {case_patch_output_dir} using component mask: {comp_image_path}")
            bag_involved(case_patch_output_dir, comp_image_path) # Moves patches from case_patch_output_dir to c1, c2... subfolders

            # Label projection for each bag, based on summary mask
            for i in range(1, 8): # c1 to c7
                bag_folder = os.path.join(case_patch_output_dir, f'c{i}')
                if os.path.exists(bag_folder) and os.listdir(bag_folder): # Check if bag folder exists and is not empty
                    print(f"    Performing label projection for bag: {bag_folder} using summary mask: {mask_image_path}")
                    label_projection(bag_folder, mask_image_path) # Moves patches from cX to cX/0, cX/1...
                else:
                    # print(f"    Skipping label projection for bag {bag_folder} as it's empty or non-existent.")
                    pass
            print(f"  Finished processing for SVS file: {svs_filename}")
        
        print(f"Finished processing case: {svs_case_folder_name}")
        print(f"  Cleaning empty folders in {case_patch_output_dir}")
        delete_empty_folders(case_patch_output_dir) # Clean empty cX/N folders and then empty cX folders

    print(f"All cases processed. Final cleanup of empty folders in main output directory: {args.output_dir}")
    delete_empty_folders(args.output_dir) # Clean any empty case folders in the main output dir
    print("Script finished.")

if __name__ == "__main__":
    main()