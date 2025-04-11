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
    ''' Коррекция размеров изображения если заданы new_width и new_height '''
    
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
    
def image3d_processing(image_name, image, depth):
    ''' Функция создания 3D на основе исходного изображения и его карты глубины '''
    
    # Если размеры исходного кадра и карты глубины не совпадают, глубина масштабируется до размеров изображения
    if image.shape[:2] != depth.shape[:2]:
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Нормализация глубины
    depth = depth.astype(np.float32) / 255.0

    # Создание параллакса
    height, width, _ = image.shape
    parallax = (depth * PARALLAX_SCALE)

    # Координаты пикселей
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    # Вычисление смещений
    shift_left = np.clip(x + parallax.astype(np.float32), 0, width - 1)
    shift_right = np.clip(x - parallax.astype(np.float32), 0, width - 1)

    # Применение смещений с cv2.remap
    left_image = cv2.remap(image, shift_left, y, interpolation=INTERPOLATION_TYPE)
    right_image = cv2.remap(image, shift_right, y, interpolation=INTERPOLATION_TYPE)
    
    if new_width and new_height:
        left_image, right_image = image_size_correction(height, width, left_image, right_image)
        # Меняем значения исходных размеров изображений на new_height и new_width для корректного склеивания ниже
        height = new_height
        width = new_width
    
    # Объединяем левое и правое изображения в общее 3D
    if TYPE3D == "HSBS":
        # Сужение ширины изображений, чтобы сделать из них одно с общей шириной
        left_image_resized = cv2.resize(left_image, (width // 2, height), interpolation=cv2.INTER_AREA)
        right_image_resized = cv2.resize(right_image, (width // 2, height), interpolation=cv2.INTER_AREA)
        # Склейка изображений в одно
        if LEFT_RIGHT == "LEFT":
            image3d = np.hstack((left_image_resized, right_image_resized))
        elif LEFT_RIGHT == "RIGHT":
            image3d = np.hstack((right_image_resized, left_image_resized))
    elif TYPE3D == "HOU":
        # Сужение высоты изображений, чтобы сделать из них одно с общей высотой
        left_image_resized = cv2.resize(left_image, (width, height // 2), interpolation=cv2.INTER_AREA)
        right_image_resized = cv2.resize(right_image, (width, height // 2), interpolation=cv2.INTER_AREA)
        # Склейка изображений в одно
        if LEFT_RIGHT == "LEFT":
            image3d = np.vstack((left_image_resized, right_image_resized))
        elif LEFT_RIGHT == "RIGHT":
            image3d = np.vstack((right_image_resized, left_image_resized))
    elif TYPE3D == "FSBS":
        # Склейка изображений в одно
        if LEFT_RIGHT == "LEFT":
            image3d = np.hstack((left_image, right_image))
        elif LEFT_RIGHT == "RIGHT":
            image3d = np.hstack((right_image, left_image))
    elif TYPE3D == "FOU":
        # Склейка изображений в одно
        if LEFT_RIGHT == "LEFT":
            image3d = np.vstack((left_image, right_image))
        elif LEFT_RIGHT == "RIGHT":
            image3d = np.vstack((right_image, left_image))

    # Сохранение 3D изображения
    output_path = os.path.join(output_dir, f'{image_name}_3d.jpg')
    cv2.imwrite(output_path, image3d, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    

# НАЧАЛО ОБРАБОТКИ
# Извлекаем имя изображения для последующего сохранения 3D изображения
image_name = os.path.splitext(os.path.basename(image_path))[0]

# Загрузка изображения и его карты глубины
image = cv2.imread(image_path)  # Исходное изображение
depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)  # Карта глубины


# ЗАПУСК ОБРАБОТЧИКА
image3d_processing(image_name, image, depth)

print("ГОТОВО.")