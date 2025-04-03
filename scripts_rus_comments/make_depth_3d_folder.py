import os
import subprocess
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value
import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2


# ПАРАМЕТРЫ ОБЩИЕ
# Папка с исходными кадрами
frames_dir = ""

# Получаем имя папки с исходными фреймами, чтобы создать папку для 3D фреймов
frames_dir_name = os.path.basename(os.path.normpath(frames_dir))
images3d_dir = os.path.join(os.path.dirname(frames_dir), f"{frames_dir_name}_3d")
os.makedirs(images3d_dir, exist_ok=True)

# Получаем список всех файлов в директории
all_frames_in_directory = [file_name for file_name in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, file_name))]

frame_counter = Value('i', 0) # Счетчик для именования кадров
threads_count = Value('i', 0) # Счетчик текущих потоков, чтобы не выходить за пределы max_threads

chunk_size = 1000  # Количество файлов на один поток
max_threads = 3 # Максимальное количество потоков

# Устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


''' КОММЕНТАРИИ К ПАРАМЕТРАМ 3D
## PARALLAX_SCALE:
Значение паралакса в пикселях, на сколько максимум пикселей будут смещаться дальние пиксели относительно ближних.
Рекомендуется от 10 до 20.

## INTERPOLATION_TYPE:
INTER_NEAREST - ближайший сосед, быстрая и простая интерполяция, не самая качественная
INTER_AREA - лучше подходит при уменьшении изображений, в данном случае не будем рассматривать
INTER_LINEAR - билинейная интерполяция, баланс качества и скорости, самый оптимальный вариант
INTER_CUBIC - бикубическая интерполяция, считается качественнее билинейной, но занимает немного больше времени
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
INTERPOLATION_TYPE = cv2.INTER_LINEAR
TYPE3D = "HOU"  # HSBS, FSBS, HOU, FOU
LEFT_RIGHT = "LEFT"  # LEFT or RIGHT

# 0 - если не нужно менять размеры полученного изображения
new_width = 0
new_height = 0

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


def image_size_correction(current_height, current_width, left_image, right_image):
    ''' Коррекция размеров изображения если заданы new_width и new_height '''
    
    # Вычисляем смещения для центрирования
    top = (new_height - current_height) // 2
    bottom = new_height - current_height - top
    left = (new_width - current_width) // 2
    right = new_width - current_width - left
    
    # Создаем черное полотно нужного размера
    new_left_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_right_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Размещаем изображение на черном фоне
    new_left_image[top:top + current_height, left:left + current_width] = left_image
    new_right_image[top:top + current_height, left:left + current_width] = right_image
    
    return new_left_image, new_right_image
            
def depth_processing(frame_name, frame_path):
    ''' Функция создания карты глубины для изображения '''
    
    # Загрузка изображения в память
    raw_img = cv2.imread(frame_path)

    # Вычисление глубины
    with torch.no_grad():
        depth = model_depth.infer_image(raw_img) # HxW raw depth map in numpy
        
    # Нормализация значений глубины от 0 до 255
    depth_normalized = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)

    return depth_normalized

def image3d_processing(frame_name, frame_path, depth):
    ''' Функция создания 3D на основе исходного изображения и его карты глубины '''
    
    # Загрузка изображения в память
    image = cv2.imread(frame_path)  # Исходное изображение

    # Нормализация глубины
    depth = depth.astype(np.float32) / 255.0  # Преобразуем в диапазон [0, 1]

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
    
    # Меняем размер кадра если заданы new_width и new_height
    if new_width != 0 and new_height != 0:
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
    output_image3d_path = os.path.join(images3d_dir, f'{frame_name}.jpg')
    cv2.imwrite(output_image3d_path, image3d, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
def extract_frames(start_frame, end_frame):
    ''' Функция разбивки файлов изображений по чанкам исходя из chunk_size '''
    
    frames_to_process = end_frame - start_frame + 1
    
    with frame_counter.get_lock():
        start_counter = frame_counter.value
        frame_counter.value += frames_to_process
        
    extracted_frames = []  # Список для хранения путей к файлам
    
    # Получаем словарь со списком файлов в размере чанка
    chunk_files = all_frames_in_directory[start_frame:end_frame+1]  # end_frame включительно
    extracted_frames = [os.path.join(frames_dir, file_name) for file_name in chunk_files]
    
    return extracted_frames

def chunk_processing(extracted_frames):
    ''' Старт обработки каждого заполненного чанка '''
    
    # Обработка depth_processing и image3d_processing
    for frame_path in extracted_frames:
        # Проверяем, что это файл, а не папка
        if not os.path.isfile(frame_path):
            continue
        
        # Извлекаем имя изображения для последующего сохранения 3D изображения
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]

        # Обработка depth_processing
        depth = depth_processing(frame_name, frame_path)

        # Обработка image3d_processing и сохранение результата
        image3d_processing(frame_name, frame_path, depth)

        # Удаление исходного файла
        os.remove(frame_path)
        
    with threads_count.get_lock():
        threads_count.value = max(1, threads_count.value - 1) # Уменьшение счетчика после завершения текущего потока
    
def run_processing():
    ''' Глобальная функция старта обработки с учетом многопоточности'''
    
    # Получение количества файлов в папке с исходниками
    total_frames = len([f for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f))])
                        
    # Управление потоками
    if total_frames:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            for start_frame in range(0, total_frames, chunk_size):
                while True:
                    with threads_count.get_lock():
                        if threads_count.value < max_threads:
                            threads_count.value += 1
                            break
                            
                    time.sleep(5) # Пауза перед повторной проверкой на количество работающих потоков

                end_frame = min(start_frame + chunk_size - 1, total_frames - 1)
                extracted_frames = extract_frames(start_frame, end_frame)
                future = executor.submit(chunk_processing, extracted_frames)
                futures.append(future)
            
            # Ожидаем завершения задач
            for future in futures:
                future.result()


# ЗАПУСК ОБРАБОТЧИКА
run_processing()


# Выгружаем модель
del model_depth
torch.cuda.empty_cache()