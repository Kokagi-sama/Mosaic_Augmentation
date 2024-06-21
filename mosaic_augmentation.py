import cv2
import numpy as np
import random
import os
import glob
import xml.etree.ElementTree as ET

def print_boundary():
    """Print a boundary within terminal for ease of sight."""
    print("-----------------------------------------------------------------------------------------------------")
    return None
    
def load_image(image_path):
    """Load an image from the given file path."""
    return cv2.imread(image_path)

def resize_image(image, target_size):
    """Resize the given image to the target size."""
    return cv2.resize(image, target_size)

def compute_coords(x_ratio, y_ratio, output_size):
    """Compute x_coords and y_coords based on partition ratios and output size."""
    if x_ratio >= 1.0 or y_ratio >= 1.0 or x_ratio <= 0 or y_ratio <= 0:
        print("\nInvalid ratio. Ratios should be between 0.0 and 1.0")
        return None, None
    
    # Calculate the boundary for x and y
    x_bound = int(x_ratio * output_size)
    y_bound = int(y_ratio * output_size)
    
    # Define coordinates for four quadrants
    x_coords = [(0, x_bound), (x_bound, output_size), (0, x_bound), (x_bound, output_size)]
    y_coords = [(0, y_bound), (0, y_bound), (y_bound, output_size), (y_bound, output_size)]
           
    return x_coords, y_coords

def crop_random_region(image, x_start, x_end, y_start, y_end):
    """Crop a random region from the image based on given start and end coordinates."""
    # Retrieve the original height and width of the image
    h, w, _ = image.shape
    
    # Calculate width and height of the region to crop
    region_width = x_end - x_start
    region_height = y_end - y_start
    
    # Ensure the region is within bounds
    if region_width <= 0 or region_height <= 0 or x_start < 0 or x_end > w or y_start < 0 or y_end > h:
        print("\nInvalid region coordinates. Please check x_start, x_end, y_start, y_end.")
        return None
    
    # Randomly select a position within the region
    start_x = random.randint(0, w - region_width)  # Ensure start_x within image bounds
    start_y = random.randint(0, h - region_height)  # Ensure start_y within image bounds
    end_x = start_x + region_width
    end_y = start_y + region_height
    
    # Crop the region from the image
    cropped_region = image[start_y:end_y, start_x:end_x]
    
    return cropped_region

def draw_bounding_box(image, xmin, ymin, xmax, ymax):
    """Draw a bounding box on the image."""
    color = (0, 255, 0)  # Green color for the bounding box
    thickness = 2
    print(f"Bounding box coordinates: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
    return cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    
def mosaic_augmentation(image_paths, x_coords, y_coords, output_size=416):
    """Create a mosaic image with variable quadrant sizes by cropping specific regions and drawing bounding boxes."""
    # Create an empty canvas for the mosaic
    mosaic_image = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    
    # Iterate over each image path and its corresponding quadrant coordinates
    for idx, image_path in enumerate(image_paths):
        # Load the original image
        original_image = load_image(image_path)
        
        # Check if image loaded successfully
        if original_image is None:
            print(f"\nFailed to load image: {image_path}")
            continue
        
        # Get the coordinates for placing the image
        x_start, x_end = x_coords[idx]
        y_start, y_end = y_coords[idx]
        
        # Find corresponding XML annotation file
        xml_path = os.path.splitext(image_path)[0] + '.xml'
        
        if os.path.exists(xml_path):
            # Load annotations from XML file
            annotations = parse_xml_annotations(xml_path)
            
            # Draw bounding boxes from annotations
            for annotation in annotations:
                xmin = int(annotation['xmin'])
                xmax = int(annotation['xmax'])
                ymin = int(annotation['ymin'])
                ymax = int(annotation['ymax'])
                original_image = draw_bounding_box(original_image, xmin, ymin, xmax, ymax)
                print(f"Object xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")

        # Resize the image to match output_size if necessary
        if original_image.shape[0] != output_size or original_image.shape[1] != output_size:
            original_image = resize_image(original_image, (output_size, output_size))
        
        # Crop a random region from the original image within its quadrant
        cropped_region = crop_random_region(original_image, x_start, x_end, y_start, y_end)
        
        if cropped_region is None:
            print(f"\nFailed to crop region from image: {image_path}")
            continue
        
        # Place the cropped region in the mosaic canvas
        mosaic_image[y_start:y_end, x_start:x_end] = cropped_region
    
    return mosaic_image


def parse_xml_annotations(xml_path):
    """Parse XML annotations from Roboflow format."""
    try:
        print(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        annotations = []
        
        for obj in root.findall('object'):
            xmin = float(obj.find('bndbox/xmin').text)
            ymin = float(obj.find('bndbox/ymin').text)
            xmax = float(obj.find('bndbox/xmax').text)
            ymax = float(obj.find('bndbox/ymax').text)
            
            annotations.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
            
        return annotations
    except FileNotFoundError:
        print(f"XML file not found: {xml_path}")
        return None
    except Exception as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return None

def display_image(image):
    """Display the given image using OpenCV."""
    print("\nMosaic Augmentation Final Output Image")
    cv2.imshow('Mosaic Augmentation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Directory containing JPEG images and XML annotations
image_directory = 'images/'

# Find all JPG files in the directory
image_paths = glob.glob(os.path.join(image_directory, '*.jpg'))

# Make sure output_size is a natural number, N >= 0
output_size = -1
while output_size < 0:
    output_size = int(input("\nInput the output size, e.g. 416 will result in final output being 416x416 image: "))
    if output_size <= 1:
        print("\nInvalid input. Please re-enter a natural number, N >=0.")
    
print_boundary()

# Make sure x_ratio is within its bound, 0.0 < x < 1.0
x_ratio = -1
while x_ratio <= 0 or x_ratio >= 1:
    x_ratio = float(input("\nInput the partition ratio of horizontal plane/x plane (0.0 < x <1.0)\nE.g. x_ratio = 0.3 will result in 1st and 3rd quadrant occupying 0.3 of the final image and 2nd and 4th quadrant occupying 0.7 of the rest of the final image.: "))
    if x_ratio <= 0 or x_ratio >= 1:
        print("\nInvalid input. Please re-enter a ratio between 0.0 and 1.0.")

print_boundary()

# Make sure y_ratio is within its bound, 0.0 < y < 1.0
y_ratio = -1
while y_ratio <= 0 or y_ratio >= 1:
    y_ratio = float(input("\nInput the partition ratio of vertical plane/y plane (0.0 < y < 1.0)\nE.g. y_ratio = 0.4 will result in 1st and 2nd quadrant occupying 0.4 of the final image and 3rd and 4th quadrant occupying 0.6 of the rest of the final image.: "))
    if y_ratio <= 0 or y_ratio >= 1:
        print("\nInvalid input. Please re-enter a ratio between 0.0 and 1.0.")

# Compute x_coords and y_coords based on ratios and output_size
x_coords, y_coords = compute_coords(x_ratio, y_ratio, output_size)

# Check if compute_coords failed
if x_coords is None or y_coords is None:
    print("\nFailed to compute coordinates for mosaic. Exiting.")
else:
    # Generate mosaic image using computed coordinates
    mosaic_image = mosaic_augmentation(image_paths, x_coords, y_coords, output_size)
    
    display_image(mosaic_image)
