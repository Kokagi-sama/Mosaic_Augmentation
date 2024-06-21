import cv2
import numpy as np
import random
import os
import glob


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

def mosaic_augmentation(image_paths, x_coords, y_coords, output_size=416):
    """Create a mosaic image with variable quadrant sizes by cropping specific regions."""
    # Create an empty canvas for the mosaic
    mosaic_image = np.zeros((output_size, output_size, 3), dtype=np.uint8)
            
    # Iterate over each image path and its corresponding quadrant coordinates
    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        
        # Resize the image to match output_size
        if image.shape[0] != output_size or image.shape[1] != output_size:
            image = resize_image(image, (output_size, output_size))
        # Get the coordinates for placing the image
        x_start, x_end = x_coords[idx]
        y_start, y_end = y_coords[idx]
        
        # Crop a random region from the image within its quadrant
        cropped_region = crop_random_region(image, x_start, x_end, y_start, y_end)
        
        if cropped_region is None:
            print(f"\nFailed to crop region from image: {image_path}")
            continue
        
        # Resize cropped region to fit exactly within its quadrant in the mosaic canvas
        resized_img = cv2.resize(cropped_region, (x_end - x_start, y_end - y_start))
        
        # Place the resized image in the mosaic canvas
        mosaic_image[y_start:y_end, x_start:x_end] = resized_img
    
    return mosaic_image

def display_image(image):
    """Display the given image using OpenCV."""
    print("\nMosaic Augmentation Final Output Image")
    cv2.imshow('Mosaic Augmentation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Directory containing JPEG images
image_directory = 'images/'

# Find all JPG files in the directory (Ideally 4 because the mosaic augmentation technique requires four images)
image_paths = glob.glob(os.path.join(image_directory, '*.jpg'))

# Make sure output_size is natural number, N >= 0
output_size = -1
while output_size < 0:
    output_size = int(input("\nInput the output size, e.g. 416 will result in final output being 416x416 image: "))
    if output_size <= 1:
        print("\nInvalid input. Please re-enter a natural number, N >=0.")
    
print_boundary()

# Make sure x_ratio is within its bound, 0.0 < x < 1.0
x_ratio = -1
while x_ratio <= 0 or x_ratio >= 1:
    x_ratio = float(input("\nInput the partition ratio of horizontal plane/x plane (0.0 < x < 1.0): "))
    if x_ratio <= 0 or x_ratio >= 1:
        print("\nInvalid input. Please re-enter a ratio between 0.0 and 1.0.")

print_boundary()

# Make sure y_ratio is within its bound, 0.0 < y < 1.0
y_ratio = -1
while y_ratio <= 0 or y_ratio >= 1:
    y_ratio = float(input("\nInput the partition ratio of vertical plane/y plane (0.0 < y < 1.0): "))
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

    print_boundary()
    # Display the final amalgamated mosaic image
    display_image(mosaic_image)