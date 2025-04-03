import os
import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2


# GENERAL OPTIONS
# Source file path
image_path = ""

# Folder to save result
output_dir = ""

# Computing device
device = torch.device('cuda')


# MODEL OPTIONS
# Path to the folder with models, specify without a slash at the end, for example: "/home/user/DepthAnythingV2/models"
depth_model_dir = ""

model_depth_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitl' # 'vitl', 'vitb', 'vits'

model_depth = DepthAnythingV2(**model_depth_configs[encoder])
model_depth.load_state_dict(torch.load(f'{depth_model_dir}/depth_anything_v2_{encoder}.pth', weights_only=True, map_location=device))
model_depth = model_depth.to(device).eval()


# START PROCESSING
# Loading the image
raw_img = cv2.imread(image_path)

# Extract the image name to save the depth map later
image_name = os.path.splitext(os.path.basename(image_path))[0]

# Depth calculation
with torch.no_grad():
    depth = model_depth.infer_image(raw_img)
    
# Normalization of depth values from 0 to 255
depth_normalized = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX)
depth_normalized = depth_normalized.astype(np.uint8)

# Saving the depth map
output_path = os.path.join(output_dir, f'{image_name}_depth.png')
cv2.imwrite(output_path, depth_normalized)


# OPTIONAL: SAVE DEPTH MAP IN COLOR
depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

# Saving the depth map in color
output_path = os.path.join(output_dir, f'{image_name}_depth_color.png')
cv2.imwrite(output_path, depth_colored)


print("DONE.")


# Delete the model from memory and clear the Cuda cache
del model_depth
torch.cuda.empty_cache()