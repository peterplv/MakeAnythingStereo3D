import os
import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2


# ПАРАМЕТРЫ ОБЩИЕ
# Путь к исходному файлу
image_path = ""

# В какую папку сохранить результат
output_dir = ""

# Устройство для вычислений
device = torch.device('cuda')


# ПАРАМЕТРЫ МОДЕЛИ
# Путь к папке с моделями, указывать без слеша на конце, например: "/home/user/DepthAnythingV2/models"
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


# НАЧАЛО ОБРАБОТКИ
# Загрузка изображения
raw_img = cv2.imread(image_path)

# Извлекаем имя изображения для последующего сохранения карты глубины
image_name = os.path.splitext(os.path.basename(image_path))[0]

# Вычисление глубины
with torch.no_grad():
    depth = model_depth.infer_image(raw_img)
    
# Нормализация глубины перед сохранением в файл
depth_normalized = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Сохранение карты глубины
output_path = os.path.join(output_dir, f'{image_name}_depth.png')
cv2.imwrite(output_path, depth_normalized)


# ОПЦИОНАЛЬНО: СОХРАНЯЕМ КАРТУ ГЛУБИНЫ В ЦВЕТЕ
depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

# Сохранение карты глубины в цвете
output_path = os.path.join(output_dir, f'{image_name}_depth_color.png')
cv2.imwrite(output_path, depth_colored)


print("ГОТОВО.")


# Удаляем модель из памяти и очищаем кеш Cuda
del model_depth
torch.cuda.empty_cache()
