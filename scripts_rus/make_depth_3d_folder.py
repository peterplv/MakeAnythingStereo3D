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
device = torch.device('cuda')


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
PARALLAX_SCALE = 15  # Максимальное значение параллакса в пикселях, рекомендуется от 10 до 20
INTERPOLATION_TYPE = cv2.INTER_LINEAR
TYPE3D = "FSBS"  # HSBS, FSBS, HOU, FOU
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
    left = (new_width - current_width) // 2
    
    # Создаем черный холст нужного размера
    new_left_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_right_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Размещаем изображение на черном фоне
    new_left_image[top:top + current_height, left:left + current_width] = left_image
    new_right_image[top:top + current_height, left:left + current_width] = right_image
    
    return new_left_image, new_right_image
            
def depth_processing(image):
    ''' Функция создания карты глубины для изображения '''
    
    # Вычисление глубины
    with torch.no_grad():
        depth = model_depth.infer_image(image)
        
    # Нормализация глубины
    depth_normalized = depth / depth.max()

    return depth_normalized

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
    ''' Функция объединения изображений стереопары в единое 3D изображение '''
    
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
    
    for frame_path in extracted_frames:
        # Проверяем, что это файл, а не папка
        if not os.path.isfile(frame_path):
            continue
        
        # Извлекаем имя изображения для последующего сохранения 3D изображения
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]
        
        # Загрузка изображения
        image = cv2.imread(frame_path)

        # Запуск depth_processing и получение карты глубины
        depth = depth_processing(image)

        # Запуск image3d_processing и получение двух изображений стереопары
        left_image, right_image, height, width = image3d_processing(image, depth)

        # Объединение стереопары в общее 3D изображение
        image3d = image3d_combining(left_image, right_image, height, width)
        
        # Сохранение 3D изображения
        output_image3d_path = os.path.join(images3d_dir, f'{frame_name}.jpg')
        cv2.imwrite(output_image3d_path, image3d, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

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

print("ГОТОВО.")


# Выгружаем модель
del model_depth
torch.cuda.empty_cache()
