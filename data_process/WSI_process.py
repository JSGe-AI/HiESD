
import os
from openslide import OpenSlide
from PIL import Image
import numpy as np
import shutil


def save_svs_patches(svs_path, output_dir,patch_size=512):
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


            patch_filename = f"{x}_{y}_512.png"

            patch.save(os.path.join(output_dir, patch_filename))


    slide.close()

def delete_backgrounds(path):

    image_files = [file for file in os.listdir(path) if file.endswith('.jpg') or file.endswith('.png')]


    for image_file in image_files:

        image_path = os.path.join(path, image_file)

         
        image = Image.open(image_path).convert('L')   
        image = image.point(lambda x: 255 if x > 128 else 0)   

         
        white_pixels = sum(pixel == 255 for pixel in image.getdata())
        total_pixels = image.width * image.height

         
        white_ratio = white_pixels / total_pixels

         
        if white_ratio > 0.75:
            os.remove(image_path)
            print(f"Deleted {image_file} due to high white ratio.")

def process_images(directory):   
     
    image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.png') or file.endswith('.jpg')]

    for image_file in image_files:
         
        image = Image.open(image_file).convert('RGB')
         
        pixels = image.load()

         
        for i in range(image.width):
            for j in range(image.height):
                 
                r, g, b = pixels[i, j]

                 
                if (r, g, b) == (1, 1, 1):
                    pixels[i, j] = (255, 0, 0)
                elif (r, g, b) == (2, 2, 2):
                    pixels[i, j] = (0, 255, 0)
                elif (r, g, b) == (3, 3, 3):
                    pixels[i, j] = (0, 0, 255)
                elif (r, g, b) == (4, 4, 4):
                    pixels[i, j] = (255, 0, 255)
                elif (r, g, b) == (5, 5, 5):
                    pixels[i, j] = (255, 192, 203)
                elif (r, g, b) == (6, 6, 6):
                    pixels[i, j] = (255, 255, 0)
                elif (r, g, b) == (7, 7, 7):
                    pixels[i, j] = (218, 112, 214)
                elif (r, g, b) == (8, 8, 8):
                    pixels[i, j] = (25, 25, 112)

         
        image.save(image_file)


def bag_involved(patch_directory,comp_image_path):

    categories = {
        '0': (1, 1, 1),
        '1': (2, 2, 2),
        '2': (3, 3, 3),
        '3': (4, 4, 4),
        '4': (5, 5, 5),
        '5': (6, 6, 6),
        '6': (7, 7, 7),
    }
    
      
    for i in range(7):
        folder_path = os.path.join(patch_directory, f"c{i+1}")
        os.makedirs(folder_path, exist_ok=True)

    bag_1 = os.path.join(patch_directory,'c1')   
    bag_2 = os.path.join(patch_directory,'c2')   
    bag_3 = os.path.join(patch_directory,'c3')   
    bag_4 = os.path.join(patch_directory,'c4')   
    bag_5 = os.path.join(patch_directory,'c5')  
    bag_6 = os.path.join(patch_directory,'c6')   
    bag_7 = os.path.join(patch_directory,'c7')  

   
     
    for filename in os.listdir(patch_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            name_parts = filename.split('_')
            if len(name_parts) == 3:
                x = int(name_parts[0])
                y = int(name_parts[1].split('.')[0])

                
                 
                target_x = x // 32
                target_y = y // 32

                 
                patch_image_path = os.path.join(patch_directory, filename)
                
                 
                target_image = Image.open(comp_image_path).convert('RGB')
                 
                 
                target_width, target_height = target_image.size

                 
                if target_x < target_width and target_y < target_height:
                     
                    start_x = target_x 
                    start_y = target_y 
                    end_x = start_x + 16
                    end_y = start_y + 16
                    
                    roi_image = target_image.crop((start_x, start_y, end_x, end_y))

                    color_counts = {category: 0 for category in categories}

                    for category, color in categories.items():
                        r, g, b = color
                        mask = np.array(roi_image) == [r, g, b]
                        color_counts[category] = np.sum(mask)

                    most_common_category = max(color_counts, key=color_counts.get)
                    most_common_color = categories[most_common_category]

            
                    if most_common_color == (1, 1, 1): 
                         
                        new_image_path = os.path.join(bag_1, filename)
                        shutil.move(patch_image_path, new_image_path)
                    elif most_common_color == (2, 2, 2):
                         
                        new_image_path = os.path.join(bag_2, filename)
                        shutil.move(patch_image_path, new_image_path)
                    elif most_common_color == (3, 3, 3):
                         
                        new_image_path = os.path.join(bag_3, filename)
                        shutil.move(patch_image_path, new_image_path)
                    elif most_common_color == (4, 4, 4):
                         
                        new_image_path = os.path.join(bag_4, filename)
                        shutil.move(patch_image_path, new_image_path)
                    elif most_common_color == (5, 5, 5):
                         
                        new_image_path = os.path.join(bag_5, filename)
                        shutil.move(patch_image_path, new_image_path)
                    elif most_common_color == (6, 6, 6):
                         
                        new_image_path = os.path.join(bag_6, filename)
                        shutil.move(patch_image_path, new_image_path)
                    elif most_common_color == (7, 7, 7):
                         
                        new_image_path = os.path.join(bag_7, filename)
                        shutil.move(patch_image_path, new_image_path)


def label_projection(bag_dic,mask_dic):
     
     
    categories = {
        '1': (139, 0, 0),
        '2': (255, 0, 255),
        '3': (128, 0, 128),
        '4': (75, 0, 130),
        '5': (138, 43, 226),
        '6': (0, 0, 255),
        '7': (65, 105, 225),
        '8': (70, 130, 180),
        '9': (0, 255, 255),
        '10': (0, 255, 0),
        '11': (0, 128, 0),
        '12': (255, 255, 0),
        '13': (255, 215, 0),
        '14': (255, 165, 0),
        '15': (255, 0, 0),
    }
    for i in range(16):
        folder_path = os.path.join(bag_dic, f"{i}")
        os.makedirs(folder_path, exist_ok=True)
    for patch_name in os.listdir(bag_dic):
        if patch_name.endswith('.jpg') or patch_name.endswith('.png'):
            name_parts = patch_name.split('_')
            if len(name_parts) == 3:
                x = int(name_parts[0])
                y = int(name_parts[1].split('.')[0])

                 
                target_x = x // 32
                target_y = y // 32

                 
                patch_image_path = os.path.join(bag_dic, patch_name)
                 

                 
                target_image = Image.open(mask_dic).convert('RGB')
                 
                 
                target_width, target_height = target_image.size

                 
                if target_x < target_width and target_y < target_height:
                     
                    start_x = target_x
                    start_y = target_y
                    end_x = start_x + 16
                    end_y = start_y + 16

                    roi_image = target_image.crop((start_x, start_y, end_x, end_y))

                    roi_image_array = np.array(roi_image)
                     
                    black_pixel_count = np.sum(np.all(roi_image_array == [0, 0, 0], axis=2))

                    color_counts = {category: 0 for category in categories}

                    for category, color in categories.items():
                        r, g, b = color
                        mask = np.array(roi_image) == [r, g, b]
                        color_counts[category] = np.sum(mask)

                    most_common_category = max(color_counts, key=color_counts.get)
                    most_common_color = categories[most_common_category]
                    
                     
                    roi_width, roi_height = roi_image.size
                    total_pixels = roi_width * roi_height
                    black_pixel_ratio = black_pixel_count / total_pixels
                    
                    
                    if black_pixel_ratio > 0.9:
                        des_path = os.path.join(bag_dic, "0")
                        shutil.move(patch_image_path, des_path)     
                    elif most_common_color == (139, 0, 0):
                        des_path = os.path.join(bag_dic,"1")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (255, 0, 255):
                        des_path = os.path.join(bag_dic,"2")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (128, 0, 128):
                        des_path = os.path.join(bag_dic,"3")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (75, 0, 130):
                        des_path = os.path.join(bag_dic, "4")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (138, 43, 226):
                        des_path = os.path.join(bag_dic, "5")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (0, 0, 255):
                        des_path = os.path.join(bag_dic,"6")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (65, 105, 225):
                        des_path = os.path.join(bag_dic, "7")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (70, 130, 180):
                        des_path = os.path.join(bag_dic, "8")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (0, 255, 255):
                        des_path = os.path.join(bag_dic, "9")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (0, 255, 0):
                        des_path = os.path.join(bag_dic,"10")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (0, 128, 0):
                        des_path = os.path.join(bag_dic, "11")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (255, 255, 0):
                        des_path = os.path.join(bag_dic,"12")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (255, 215, 0):
                        des_path = os.path.join(bag_dic,"13")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (255, 165, 0):
                        des_path = os.path.join(bag_dic, "14")
                        shutil.move(patch_image_path, des_path)
                    elif most_common_color == (255, 0, 0):
                        des_path = os.path.join(bag_dic, "15")
                        shutil.move(patch_image_path, des_path)



def delete_folders(folder):
    def delete_empty_folders(folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    print(f"delete empty files: {dir_path}")

         
        delete_empty_folders_recursive(folder_path)

    def delete_empty_folders_recursive(folder_path):
        if not os.listdir(folder_path):
            os.rmdir(folder_path)
            print(f"delete empty files: {folder_path}")
            parent_folder = os.path.dirname(folder_path)
            if parent_folder != folder_path:
                delete_empty_folders_recursive(parent_folder)

    delete_empty_folders(folder)



final_data_path ="C:/Users/52257/Desktop/code/Early_gastric_cancer/new_data/final_data/"   
svs_dir = "new_data/svs"     
mask_components_dir="new_data/ESD_XJ_components"       
summary_dir="new_data/ESD_XJ_summary"

for svs_file in os.listdir(svs_dir):
    svs_file_path = os.path.join(svs_dir,svs_file)
    patch_directory = os.path.join(final_data_path,svs_file)
    if os.path.exists(patch_directory):
        continue   
    os.mkdir(os.path.join(final_data_path,svs_file))
    
    for svs in os.listdir(svs_file_path):
        svs_tmp_dir = os.path.join(svs_file_path,svs)
        save_svs_patches(svs_tmp_dir, patch_directory)   
        delete_backgrounds(patch_directory)   
        
        comp_file_name = svs_file + '_comimg.jpg'
        comp_image_path = os.path.join(mask_components_dir,comp_file_name)   
        mask_image_path = os.path.join(summary_dir,svs_file + '.png')      

        output_folder_1 = os.path.join(patch_directory,'c1')   
        output_folder_2 = os.path.join(patch_directory,'c2')   
        output_folder_3 = os.path.join(patch_directory,'c3')   
        output_folder_4 = os.path.join(patch_directory,'c4')   
        output_folder_5 = os.path.join(patch_directory,'c5')  
        output_folder_6 = os.path.join(patch_directory,'c6')   
        output_folder_7 = os.path.join(patch_directory,'c7')  

        bag_involved(patch_directory,comp_image_path)   
        label_projection(output_folder_1,mask_image_path)
        label_projection(output_folder_2,mask_image_path)
        label_projection(output_folder_3,mask_image_path)
        label_projection(output_folder_4,mask_image_path)
        label_projection(output_folder_5,mask_image_path)
        label_projection(output_folder_6,mask_image_path)
        label_projection(output_folder_7,mask_image_path)


     
delete_folders(final_data_path)