import os
import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2


# GENERAL OPTIONS
# Path to the folder with depth generation models
depth_models_path = ""

# Source file path
image_path = ""

# Folder to save result
output_path = ""

# Computing device
device = torch.device('cuda')


# DEPTH OPTIONS
depth_models_config = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

# Selecting the DepthAnythingV2 model: vits - Small, vitb - Base, vitl - Large
encoder = "vitl" # vits, vitb, vitl

model_depth_current = os.path.join(depth_models_path, f'depth_anything_v2_{encoder}.pth')
model_depth = DepthAnythingV2(**depth_models_config[encoder])
model_depth.load_state_dict(torch.load(model_depth_current, weights_only=True, map_location=device))
model_depth = model_depth.to(device).eval()


# START PROCESSING
# Loading the image
raw_img = cv2.imread(image_path)

# Extract the image name to save the depth map later
image_name = os.path.splitext(os.path.basename(image_path))[0]

# Depth calculation
with torch.no_grad():
    depth = model_depth.infer_image(raw_img)
    
# Depth normalization before saving
depth_normalized = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Saving the depth map
output_path = os.path.join(output_path, f'{image_name}_depth.png')
cv2.imwrite(output_path, depth_normalized)

# OPTIONAL: SAVE DEPTH MAP IN COLOR
depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

# Saving the depth map in color
output_path = os.path.join(output_path, f'{image_name}_depth_color.png')
cv2.imwrite(output_path, depth_colored)


print("DONE.")


# Delete model and clear Cuda cache
del model_depth
torch.cuda.empty_cache()