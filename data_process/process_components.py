import cv2
import numpy as np
import os

def convert_to_binary(image_path, threshold):
     
    image = cv2.imread(image_path)

     
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

     
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return binary

def extract_connected_components(image):
     
     

     
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

     
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

     
    connected_components = []

     
    for label in range(1, num_labels):   
         
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

         
        component = image[top:top+height, left:left+width]

         
        connected_components.append((component, (left, top, width, height)))

    return connected_components, labels

 
threshold_value = 10

 
image_path = "C:/Users/52257/Desktop/fsdownload/summary -copy/"
 
datanames = os.listdir(image_path)
for img in datanames:
    binary_image = convert_to_binary(image_path+img, threshold_value)
     
    kernel = np.ones((20, 20), np.uint8)   
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
     
    components, com_img = extract_connected_components(dilated_image)

     
    cv2.imwrite(f"C:/Users/52257/Desktop/fsdownload/summary/{img.split('.')[0]}.jpg", com_img)