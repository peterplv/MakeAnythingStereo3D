import os
import cv2
import numpy as np


# GENERAL OPTIONS
# Source file path
image_path = ""

# Depth map path for the source image
depth_path = ""

# Folder to save result
output_dir = ""


''' 3D OPTIONS COMMENTS
## PARALLAX_SCALE:
Parallax value in pixels, by how many maximum pixels the far pixels will be shifted relative to the near pixels.
Recommended from 10 to 20.

## INTERPOLATION_TYPE:
INTER_NEAREST – Nearest-neighbor interpolation. Fastest but lowest quality (pixelated edges).
INTER_AREA – Best for image reduction (averaging pixels). Not considered in this case.
INTER_LINEAR – Bilinear interpolation (2×2 pixel neighborhood). Balanced quality and speed (recommended default).
INTER_CUBIC – Bicubic interpolation (4×4 pixel neighborhood). Higher quality than linear but slower.
INTER_LANCZOS4 – Lanczos interpolation (8×8 pixel neighborhood). Highest quality but significantly slower.

## TYPE3D:
HSBS (Half Side-by-Side) - half horizontal stereopair
FSBS (Full Side-by-Side) - full horizontal stereopair
HOU (Half Over-Under) - half vertical stereopair
FOU (Full Over-Under) - full vertical stereopair

## LEFT_RIGHT:
The order of a pair of frames in the overall 3D image, LEFT is left first, RIGHT is right first.

## new_width + new_height:
Change the resolution of the output image without warping (with black margins added).
If there's no need to change, then new_width = 0 and new_height = 0
'''

# 3D PARAMETERS
PARALLAX_SCALE = 15  # Recommended 10 to 20
INTERPOLATION_TYPE = cv2.INTER_LINEAR
TYPE3D = "FSBS"  # HSBS, FSBS, HOU, FOU
LEFT_RIGHT = "LEFT"  # LEFT or RIGHT

# 0 - if there's no need to change frame size
new_width = 0
new_height = 0


def image_size_correction(current_height, current_width, left_image, right_image):
    ''' Image size correction if new_width and new_height are set '''
    
    # Calculate offsets for centering
    top = (new_height - current_height) // 2
    left = (new_width - current_width) // 2
    
    # Create a black canvas of the desired size
    new_left_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_right_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Placing the image on a black background
    new_left_image[top:top + current_height, left:left + current_width] = left_image
    new_right_image[top:top + current_height, left:left + current_width] = right_image
    
    return new_left_image, new_right_image
    
def image3d_processing(image, depth):
    ''' The function of creating a stereo pair based on the source image and depth map '''
    
    # Image size
    height, width, _ = image.shape
    
    # Creating parallax
    parallax = depth * PARALLAX_SCALE

    # Pixel coordinates
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    # Calculation of offsets
    shift_left = np.clip(x - parallax, 0, width - 1)
    shift_right = np.clip(x + parallax, 0, width - 1)

    # Applying offsets with cv2.remap
    left_image = cv2.remap(image, shift_left, y, interpolation=INTERPOLATION_TYPE)
    right_image = cv2.remap(image, shift_right, y, interpolation=INTERPOLATION_TYPE)
    
    return left_image, right_image, height, width
    
def image3d_combining(left_image, right_image, height, width):   
    ''' Function for combining stereo pair images into a single 3D image '''
    
    # Images size correction if new_width and new_height are set
    if new_width and new_height:
        left_image, right_image = image_size_correction(height, width, left_image, right_image)
        # Change the values of the original image sizes to new_height and new_width for correct gluing below
        height = new_height
        width = new_width
    
    # Combine left and right images into a common 3D image
    if TYPE3D in ("HSBS", "HOU"):
        resize_dims = (width // 2, height) if TYPE3D == "HSBS" else (width, height // 2)
        stack_func = np.hstack if TYPE3D == "HSBS" else np.vstack

        left_resized = cv2.resize(left_image, resize_dims, interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(right_image, resize_dims, interpolation=cv2.INTER_AREA)
        
        return stack_func((left_resized, right_resized)) if LEFT_RIGHT == "LEFT" else stack_func((right_resized, left_resized))
    
    elif TYPE3D in ("FSBS", "FOU"):
        stack_func = np.hstack if TYPE3D == "FSBS" else np.vstack
        
        return stack_func((left_image, right_image)) if LEFT_RIGHT == "LEFT" else stack_func((right_image, left_image))
    

# PREPARATION
# Extract the image name to save the 3D image later on
image_name = os.path.splitext(os.path.basename(image_path))[0]

# Load image and depth map
image = cv2.imread(image_path)  # Source image
depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0  # Depth map


# START PROCESSING
# Runing image3d_processing and getting a stereo pair for the image
left_image, right_image, height, width = image3d_processing(image, depth)

# Combining stereo pair into a common 3D image
image3d = image3d_combining(left_image, right_image, height, width)

# Saving 3D image
output_path = os.path.join(output_dir, f'{image_name}_3d.jpg')
cv2.imwrite(output_path, image3d, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


print("DONE.")
