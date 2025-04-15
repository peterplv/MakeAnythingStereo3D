import os
import cv2
import numpy as np


# ПАРАМЕТРЫ ОБЩИЕ
# Путь к исходному файлу
image_path = ""

# Путь к карте глубине для исходного файла
depth_path = ""

# В какую папку сохранить 3D изображение
output_dir = ""


''' КОММЕНТАРИИ К ПАРАМЕТРАМ 3D
## PARALLAX_SCALE:
Значение параллакса в пикселях, на сколько максимум пикселей будут смещаться дальние пиксели относительно ближних.
Рекомендуется от 10 до 20.

## INTERPOLATION_TYPE:
INTER_NEAREST - ближайший сосед, быстрая и простая интерполяция, не самая качественная
INTER_AREA - лучше подходит при уменьшении изображений, в данном случае не будем рассматривать
INTER_LINEAR - билинейная интерполяция в окрестности 2x2 пикселей, баланс качества и скорости, самый оптимальный вариант
INTER_CUBIC - бикубическая интерполяция в окрестности 4x4 пикселей, считается качественнее билинейной, но занимает немного больше времени
INTER_LANCZOS4 - интерполяция Ланцоша в окрестности 8x8 пикселей, самое высокое качество, но работает существенно медленнее остальных

## TYPE3D:
HSBS (Half Side-by-Side) - половинчатая горизонтальная стереопара
FSBS (Full Side-by-Side) - полная горизонтальная стереопара
HOU (Half Over-Under) - половинчатая вертикальная стереопара
FOU (Full Over-Under) - полная вертикальная стереопара

## LEFT_RIGHT:
Порядок пары кадров в общем 3D изображении, LEFT - сначала левый, RIGHT - сначала правый.

## new_width + new_height:
Изменение разрешения выходного изображения без деформации (с добавлением черных полей).
Если не нужно менять, тогда new_width = 0 и new_height = 0
'''

# ПАРАМЕТРЫ 3D
PARALLAX_SCALE = 15  # Рекомендуется от 10 до 20
INTERPOLATION_TYPE = cv2.INTER_LINEAR
TYPE3D = "FSBS"  # HSBS, FSBS, HOU, FOU
LEFT_RIGHT = "LEFT"  # LEFT or RIGHT

# 0 - если не нужно менять размеры полученного изображения
new_width = 0
new_height = 0


def image_size_correction(current_height, current_width, left_image, right_image):
    ''' Коррекция размеров изображений если заданы new_width и new_height '''
    
    # Вычисляем смещения для центрирования
    top = (new_height - current_height) // 2
    left = (new_width - current_width) // 2
    
    # Создаем черный холст нужного размера
    new_left_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_right_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Размещаем изображение на черном фоне
    new_left_image[top:top + current_height, left:left + current_width] = left_image
    new_right_image[top:top + current_height, left:left + current_width] = right_image
    
    return new_left_image, new_right_image
    
def image3d_processing(image, depth):
    ''' Функция создания стереопары на основе исходного изображения и карты глубины '''
    
    # Размеры изображения
    height, width, _ = image.shape
    
    # Создание параллакса
    parallax = depth * PARALLAX_SCALE

    # Координаты пикселей
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    # Вычисление смещений
    shift_left = np.clip(x - parallax, 0, width - 1)
    shift_right = np.clip(x + parallax, 0, width - 1)

    # Применение смещений с cv2.remap
    left_image = cv2.remap(image, shift_left, y, interpolation=INTERPOLATION_TYPE)
    right_image = cv2.remap(image, shift_right, y, interpolation=INTERPOLATION_TYPE)
    
    return left_image, right_image, height, width
    
def image3d_combining(left_image, right_image, height, width):   
    ''' Объединение изображений стереопары в единое 3D изображение '''
    
    # Корректировка размеров изображений, если заданы new_width и new_height
    if new_width and new_height:
        left_image, right_image = image_size_correction(height, width, left_image, right_image)
        # Меняем значения исходных размеров изображений на new_height и new_width для корректного склеивания ниже
        height = new_height
        width = new_width
    
    # Объединение стереопары в 3D изображение
    if TYPE3D in ("HSBS", "HOU"):
        resize_dims = (width // 2, height) if TYPE3D == "HSBS" else (width, height // 2)
        stack_func = np.hstack if TYPE3D == "HSBS" else np.vstack

        left_resized = cv2.resize(left_image, resize_dims, interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(right_image, resize_dims, interpolation=cv2.INTER_AREA)
        
        return stack_func((left_resized, right_resized)) if LEFT_RIGHT == "LEFT" else stack_func((right_resized, left_resized))
    
    elif TYPE3D in ("FSBS", "FOU"):
        stack_func = np.hstack if TYPE3D == "FSBS" else np.vstack
        
        return stack_func((left_image, right_image)) if LEFT_RIGHT == "LEFT" else stack_func((right_image, left_image))

    
# ПОДГОТОВКА
# Извлекаем имя изображения для последующего сохранения 3D изображения
image_name = os.path.splitext(os.path.basename(image_path))[0]

# Загрузка изображения и его карты глубины
image = cv2.imread(image_path)  # Исходное изображение
depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0  # Карта глубины


# СТАРТ ОБРАБОТКИ
# Запуск image3d_processing и получение двух изображений стереопары
left_image, right_image, height, width = image3d_processing(image, depth)

# Объединение стереопары в общее 3D изображение
image3d = image3d_combining(left_image, right_image, height, width)

# Сохранение 3D изображения
output_path = os.path.join(output_dir, f'{image_name}_3d.jpg')
cv2.imwrite(output_path, image3d, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


print("ГОТОВО.")
