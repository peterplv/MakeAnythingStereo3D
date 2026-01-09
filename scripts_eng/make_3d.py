import os
import cv2
import numpy as np


# GENERAL OPTIONS
# Source file path
image_path = ""

# Depth map path for the source image
depth_path = ""

# Folder to save result
output_path = ""


''' 3D OPTIONS COMMENTS
## PARALLAX_SCALE:
Parallax value in pixels, by how many maximum pixels the far pixels will be shifted relative to the near pixels.
Recommended from 10 to 20.

## PARALLAX_METHOD:
Parallax method, 1 or 2; 1 is faster and smoother, 2 is slower but can be more accurate 3D.

## INPAINT_RADIUS:
The radius of offset filling in pixels for the image3d_processing_method2 function, in this case it is the filling of neighboring pixels on the edges of images when they are offset, is recommended from 2 to 5, the optimal value is 2-3.

## INTERPOLATION_TYPE:
INTER_NEAREST – Nearest-neighbor interpolation. Fastest but lowest quality (pixelated edges).
INTER_AREA – Best for image reduction (averaging pixels). Not considered in this case.
INTER_LINEAR – Bilinear interpolation (2×2 pixel neighborhood). Balanced quality and speed (recommended default).
INTER_CUBIC – Bicubic interpolation (4×4 pixel neighborhood). Higher quality than linear but slower.
INTER_LANCZOS4 – Lanczos interpolation (8×8 pixel neighborhood). Highest quality but significantly slower.

## TYPE3D:
HSBS (Half Side-by-Side) - half horizontal stereopair.
FSBS (Full Side-by-Side) - full horizontal stereopair.
HOU (Half Over-Under) - half vertical stereopair.
FOU (Full Over-Under) - full vertical stereopair.

## LEFT_RIGHT:
The order of a pair of frames in the overall 3D image, LEFT is left first, RIGHT is right first.

## new_width + new_height:
Change the resolution of the output image without warping (with black margins added).
If there's no need to change, then new_width = 0 and new_height = 0.
'''

# 3D OPTIONS
PARALLAX_SCALE = 15  # Recommended 10 to 20
PARALLAX_METHOD = 2  # 1 or 2
INPAINT_RADIUS  = 2  # For PARALLAX_METHOD = 2 only, recommended 2 to 5, optimum value 2-3
INTERPOLATION_TYPE = cv2.INTER_LINEAR
TYPE3D = "FSBS"  # HSBS, FSBS, HOU, FOU
LEFT_RIGHT = "LEFT"  # LEFT or RIGHT

# 0 - if there's no need to change frame size
new_width  = 0
new_height = 0


def image_size_correction(current_height, current_width, left_image, right_image):
    ''' Image size correction if new_width and new_height are set '''
    
    # Calculate offsets for centering
    top = (new_height - current_height) // 2
    left = (new_width - current_width) // 2
    
    # Create a black canvas of the desired size
    new_left_image  = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_right_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Placing the image on a black background
    new_left_image[top:top + current_height, left:left + current_width] = left_image
    new_right_image[top:top + current_height, left:left + current_width] = right_image
    
    return new_left_image, new_right_image
    
def image3d_processing_method1(image, depth, height, width):
    ''' The function of creating a stereo pair based on the source image and depth map.
        Method1: faster, contours smoother, but may be less accurate
    '''
    
    # Creating parallax
    parallax = depth * PARALLAX_SCALE

    # Pixel coordinates
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    # Calculation of offsets
    shift_left =  np.clip(x - parallax, 0, width - 1)
    shift_right = np.clip(x + parallax, 0, width - 1)

    # Applying offsets with cv2.remap
    left_image =  cv2.remap(image, shift_left,  y, interpolation=INTERPOLATION_TYPE)
    right_image = cv2.remap(image, shift_right, y, interpolation=INTERPOLATION_TYPE)
    
    return left_image, right_image

def image3d_processing_method2(image, depth, height, width):
    ''' The function of creating a stereo pair based on the source image and depth map.
        Method2: slightly slower than the first method, but can be more accurate
    '''
    
    # Calculating the value for parallax
    parallax = depth * PARALLAX_SCALE
    
    # Parallax rounding and conversion to int32
    shift = np.round(parallax).astype(np.int32)

    # Grid coordinates
    y, x = np.indices((height, width), dtype=np.int32)

    # Image preparation
    left_image  = np.zeros_like(image)
    right_image = np.zeros_like(image)

    # Left image shaping by offset coordinates
    x_src_left = x - shift
    valid_left = (x_src_left >= 0) & (x_src_left < width)
    left_image[y[valid_left], x[valid_left]] = image[y[valid_left], x_src_left[valid_left]]

    # Right image shaping by offset coordinates
    x_src_right = x + shift
    valid_right = (x_src_right >= 0) & (x_src_right < width)
    right_image[y[valid_right], x[valid_right]] = image[y[valid_right], x_src_right[valid_right]]
    
    # Missing pixel masks for inpainting
    mask_left  = (~valid_left).astype(np.uint8) * 255
    mask_right = (~valid_right).astype(np.uint8) * 255

    # Filling voids via inpainting
    left_image  = cv2.inpaint(left_image,  mask_left,  INPAINT_RADIUS, cv2.INPAINT_TELEA)
    right_image = cv2.inpaint(right_image, mask_right, INPAINT_RADIUS, cv2.INPAINT_TELEA)

    return left_image, right_image
    
def image3d_combining(left_image, right_image, height, width):   
    ''' Combining stereo pair images into a single 3D image '''
    
    # Images size correction if new_width and new_height are set
    if new_width and new_height:
        left_image, right_image = image_size_correction(height, width, left_image, right_image)
        # Change the values of the original image sizes to new_height and new_width for correct gluing below
        height = new_height
        width = new_width
        
    # Image order, left first or right first
    img1, img2 = (left_image, right_image) if LEFT_RIGHT == "LEFT" else (right_image, left_image)
    
    # Combine left and right images into a common 3D image
    if TYPE3D == "HSBS":  # Narrowing and combining images horizontally
        combined_image = np.hstack((cv2.resize(img1, (width // 2, height), interpolation=cv2.INTER_AREA),
                          cv2.resize(img2, (width // 2, height), interpolation=cv2.INTER_AREA)))
                          
    elif TYPE3D == "HOU":  # Narrowing and combining images vertically
        combined_image = np.vstack((cv2.resize(img1, (width, height // 2), interpolation=cv2.INTER_AREA),
                          cv2.resize(img2, (width, height // 2), interpolation=cv2.INTER_AREA)))
                          
    elif TYPE3D == "FSBS":  # Combining images horizontally
        combined_image = np.hstack((img1, img2))
    
    elif TYPE3D == "FOU":  # Combining images vertically
        combined_image = np.vstack((img1, img2))
    
    return combined_image
    

# PREPARATION
# Extract the image name to save the 3D image later on
image_name = os.path.splitext(os.path.basename(image_path))[0]

# Load image and depth map
image = cv2.imread(image_path)  # Source image
depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0  # Depth map

# Image size
height, width = image.shape[:2]

# START PROCESSING
# Runing image3d_processing and getting a stereo pair for the image
if PARALLAX_METHOD == 1:
    left_image, right_image = image3d_processing_method1(image, depth, height, width)
elif PARALLAX_METHOD == 2:
    left_image, right_image = image3d_processing_method2(image, depth, height, width)

# Combining stereo pair into a common 3D image
image3d = image3d_combining(left_image, right_image, height, width)

# Saving 3D image
output_path = os.path.join(output_path, f'{image_name}_3d.jpg')
cv2.imwrite(output_path, image3d, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


print("DONE.")