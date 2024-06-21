import cv2
import numpy as np
import random
import os
import glob

def load_image(image_path):
    """Load an image from the given file path."""
    return cv2.imread(image_path)

def crop_random_region(image, x_ratio, y_ratio):
    """Crop a random region from the image based on x_ratio and y_ratio."""
    h, w, _ = image.shape
    start_x = random.randint(0, int(w * (1 - x_ratio)))
    start_y = random.randint(0, int(h * (1 - y_ratio)))
    end_x = start_x + int(w * x_ratio)
    end_y = start_y + int(h * y_ratio)
    return image[start_y:end_y, start_x:end_x]

def resize_image(image, width, height):
    """Resize the image to the given width and height."""
    return cv2.resize(image, (width, height))

def mosaic_augmentation(image_paths, x_ratio, y_ratio, output_size=640):
    """Create a mosaic image with variable quadrant sizes by cropping random regions."""
    # Load and crop random regions from images
    cropped_images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        cropped_region = crop_random_region(image, x_ratio, y_ratio)
        cropped_images.append(cropped_region)

    # Create an empty canvas for the mosaic
    mosaic_image = np.zeros((output_size, output_size, 3), dtype=np.uint8)

    # Calculate starting coordinates for each quadrant
    x_offsets = [0, output_size // 2, 0, output_size // 2]
    y_offsets = [0, 0, output_size // 2, output_size // 2]

    # Place each cropped and resized image at the specified coordinates
    for img, x_offset, y_offset in zip(cropped_images, x_offsets, y_offsets):
        h, w, _ = img.shape
        # Resize the cropped image to fit within its quadrant
        resized_img = resize_image(img, output_size // 2, output_size // 2)
        h_resized, w_resized, _ = resized_img.shape
        # Ensure the resized image fits within the mosaic boundaries
        mosaic_image[y_offset:y_offset + h_resized, x_offset:x_offset + w_resized] = resized_img

    return mosaic_image

def display_image(image):
    """Display the given image using OpenCV."""
    cv2.imshow('Variable Quadrant Size Mosaic', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Directory containing JPEG images
image_directory = 'images/'

# Find all JPEG files in the directory
image_paths = glob.glob(os.path.join(image_directory, '*.jpg'))

# Print the list of image paths (for verification)
print("Found JPEG images:")
for image_path in image_paths:
    print(image_path)

# Adjust x_ratio and y_ratio to control the amount of each image to be cropped
x_ratio = 0.3  # Example: x_ratio controls the horizontal partitioning
y_ratio = 0.9  # Example: y_ratio controls the vertical partitioning

mosaic_image = mosaic_augmentation(image_paths, x_ratio, y_ratio)
display_image(mosaic_image)
