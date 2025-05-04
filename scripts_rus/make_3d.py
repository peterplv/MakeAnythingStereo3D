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

## PARALLAX_METHOD:
Метод параллакса, 1 или 2; 1 - быстрее и более сглаженный, 2 - медленнее, но может быть точнее 3D

## INPAINT_RADIUS:
Радиус заполнения смещений в пикселях для функции image3d_processing_method2, в данном случае это заполнение соседними пикселями на краях изображений при их смещении, рекомендуется от 2 до 5, оптимальное значение 2-3

## INTERPOLATION_TYPE:
Тип интерполяции для функции image3d_processing_method1.
INTER_NEAREST - ближайший сосед, быстрая и простая интерполяция, не самая качественная
INTER_AREA - лучше подходит при уменьшении изображений, в данном случае не рассматривается
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
PARALLAX_SCALE = 15  # Максимальное значение параллакса в пикселях, рекомендуется от 10 до 20
PARALLAX_METHOD = 2  # 1 или 2
INPAINT_RADIUS = 2  # Рекомендуется от 2 до 5, оптимальное значение 2-3
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
    
def image3d_processing_method1(image, depth, height, width):
    ''' Функция создания стереопары на основе исходного изображения и карты глубины.
        Метод1: более быстрый, контуры более сглаженные, но может быть менее точным
    '''
    
    # Вычисление значения для параллакса
    parallax = depth * PARALLAX_SCALE
    
    # Сетка координат
    y, x = np.indices((height, width), dtype=np.float32)

    # Вычисление смещений
    shift_left = np.clip(x - parallax, 0, width - 1)
    shift_right = np.clip(x + parallax, 0, width - 1)

    # Применение смещений с cv2.remap
    left_image = cv2.remap(image, shift_left, y, interpolation=INTERPOLATION_TYPE)
    right_image = cv2.remap(image, shift_right, y, interpolation=INTERPOLATION_TYPE)

    return left_image, right_image
    
def image3d_processing_method2(image, depth, height, width):
    ''' Функция создания стереопары на основе исходного изображения и карты глубины.
        Метод2: немного медленнее первого метода, но может быть точнее.
    '''
    
    # Вычисление значения для параллакса
    parallax = depth * PARALLAX_SCALE
    
    # Округление параллакса и преобразование в int32
    shift = np.round(parallax).astype(np.int32)

    # Сетка координат
    y, x = np.indices((height, width))

    # Подготовка изображений
    left_image  = np.zeros_like(image)
    right_image = np.zeros_like(image)

    # Формирование левого изображения по смещенным координатам
    x_src_left = x - shift
    valid_left = (x_src_left >= 0) & (x_src_left < width)
    left_image[y[valid_left], x[valid_left]] = image[y[valid_left], x_src_left[valid_left]]

    # Формирование правого изображения по смещенным координатам
    x_src_right = x + shift
    valid_right = (x_src_right >= 0) & (x_src_right < width)
    right_image[y[valid_right], x[valid_right]] = image[y[valid_right], x_src_right[valid_right]]
    
    # Маски пропущенных пикселей для инпейнтинга
    mask_left  = (~valid_left).astype(np.uint8) * 255
    mask_right = (~valid_right).astype(np.uint8) * 255

    # Заполнение пустот через инпейнтинг
    left_image  = cv2.inpaint(left_image,  mask_left,  INPAINT_RADIUS,  cv2.INPAINT_TELEA)
    right_image = cv2.inpaint(right_image, mask_right, INPAINT_RADIUS, cv2.INPAINT_TELEA)

    return left_image, right_image
    
def image3d_combining(left_image, right_image, height, width):   
    ''' Объединение изображений стереопары в единое 3D изображение '''
    
    # Корректировка размеров изображений, если заданы new_width и new_height
    if new_width and new_height:
        left_image, right_image = image_size_correction(height, width, left_image, right_image)
        # Меняем значения исходных размеров изображений на new_height и new_width для корректного склеивания ниже
        height = new_height
        width = new_width
        
    # Порядок изображений, сначала левое или сначала правое
    img1, img2 = (left_image, right_image) if LEFT_RIGHT == "LEFT" else (right_image, left_image)
    
    # Объединение левого и правого изображений в единое 3D изображение
    if TYPE3D == "HSBS":  # Сужение и склейка изображений по горизонтали
        combined_image = np.hstack((cv2.resize(img1, (width // 2, height), interpolation=cv2.INTER_AREA),
                          cv2.resize(img2, (width // 2, height), interpolation=cv2.INTER_AREA)))
                          
    elif TYPE3D == "HOU":  # Сужение и склейка изображений по вертикали
        combined_image = np.vstack((cv2.resize(img1, (width, height // 2), interpolation=cv2.INTER_AREA),
                          cv2.resize(img2, (width, height // 2), interpolation=cv2.INTER_AREA)))
                          
    elif TYPE3D == "FSBS":  # Склейка изображений по горизонтали
        combined_image = np.hstack((img1, img2))
    
    elif TYPE3D == "FOU":  # Склейка изображений по вертикали
        combined_image = np.vstack((img1, img2))
    
    return combined_image


# ПОДГОТОВКА
# Извлекаем имя изображения для последующего сохранения 3D изображения
image_name = os.path.splitext(os.path.basename(image_path))[0]

# Загрузка изображения и его карты глубины
image = cv2.imread(image_path)  # Исходное изображение
depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0  # Карта глубины

# Размеры изображения
height, width = image.shape[:2]

# СТАРТ ОБРАБОТКИ
# Запуск image3d_processing и получение двух изображений стереопары
if PARALLAX_METHOD == 1:
    left_image, right_image = image3d_processing_method1(image, depth, height, width)
elif PARALLAX_METHOD == 2:
    left_image, right_image = image3d_processing_method2(image, depth, height, width)
else:
    print(f"Задайте корректный {PARALLAX_METHOD}.")

# Объединение стереопары в общее 3D изображение
image3d = image3d_combining(left_image, right_image, height, width)

# Сохранение 3D изображения
output_path = os.path.join(output_dir, f'{image_name}_3d.jpg')
cv2.imwrite(output_path, image3d, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


print("ГОТОВО.")