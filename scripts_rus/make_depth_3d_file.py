import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value
import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2


# ПАРАМЕТРЫ ОБЩИЕ
# Путь к папке с моделями генерации глубины
depth_models_path = ""

# Исходный файл
video_file_path = ""
video_file_name = os.path.splitext(os.path.basename(video_file_path))[0]

# Папка для выгрузки фреймов и папка для итоговых 3D фреймов
frames_path = os.path.join(os.path.dirname(video_file_path), f"{video_file_name}_frames")
images3d_path = os.path.join(os.path.dirname(video_file_path), f"{video_file_name}_3d")
os.makedirs(frames_path, exist_ok=True)
os.makedirs(images3d_path, exist_ok=True)

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

## PARALLAX_METHOD:
Метод параллакса, 1 или 2; 1 - быстрее и более сглаженный, 2 - медленнее, но может быть точнее 3D.

## INPAINT_RADIUS:
Радиус заполнения смещений в пикселях для функции image3d_processing_method2, в данном случае это заполнение соседними пикселями на краях изображений при их смещении, рекомендуется от 2 до 5, оптимальное значение 2-3.

## INTERPOLATION_TYPE:
Тип интерполяции для функции image3d_processing_method1.
INTER_NEAREST - ближайший сосед, быстрая и простая интерполяция, не самая качественная.
INTER_AREA - лучше подходит при уменьшении изображений, в данном случае не рассматривается.
INTER_LINEAR - билинейная интерполяция в окрестности 2x2 пикселей, баланс качества и скорости, самый оптимальный вариант.
INTER_CUBIC - бикубическая интерполяция в окрестности 4x4 пикселей, считается качественнее билинейной, но занимает немного больше времени.
INTER_LANCZOS4 - интерполяция Ланцоша в окрестности 8x8 пикселей, самое высокое качество, но работает существенно медленнее остальных.

## TYPE3D:
HSBS (Half Side-by-Side) - половинчатая горизонтальная стереопара.
FSBS (Full Side-by-Side) - полная горизонтальная стереопара.
HOU (Half Over-Under) - половинчатая вертикальная стереопара.
FOU (Full Over-Under) - полная вертикальная стереопара.

## LEFT_RIGHT:
Порядок пары кадров в общем 3D изображении, LEFT - сначала левый, RIGHT - сначала правый.

## new_width + new_height:
Изменение разрешения выходного изображения без деформации (с добавлением черных полей).
Если не нужно менять, тогда new_width = 0 и new_height = 0.
'''

# ПАРАМЕТРЫ 3D
PARALLAX_SCALE = 15  # Максимальное значение параллакса в пикселях, рекомендуется от 10 до 20
PARALLAX_METHOD = 2  # 1 или 2
INPAINT_RADIUS  = 2  # Для PARALLAX_METHOD = 2, рекомендуется от 2 до 5, оптимальное значение 2-3
INTERPOLATION_TYPE = cv2.INTER_LINEAR
TYPE3D = "FSBS"  # HSBS, FSBS, HOU, FOU
LEFT_RIGHT = "LEFT"  # LEFT or RIGHT

# 0 - если не нужно менять размеры полученного изображения
new_width  = 0
new_height = 0

depth_models_config = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

# Выбор модели DepthAnythingV2: vits - Small, vitb - Base, vitl - Large
encoder = "vitl" # vits, vitb, vitl

model_depth_current = os.path.join(depth_models_path, f'depth_anything_v2_{encoder}.pth')
model_depth = DepthAnythingV2(**depth_models_config[encoder])
model_depth.load_state_dict(torch.load(model_depth_current, weights_only=True, map_location=device))
model_depth = model_depth.to(device).eval()
 

def image_size_correction(current_height, current_width, left_image, right_image):
    ''' Коррекция размеров изображений если заданы new_width и new_height '''
    
    # Вычисляем смещения для центрирования
    top = (new_height - current_height) // 2
    left = (new_width - current_width) // 2
    
    # Создаем черный холст нужного размера
    new_left_image  = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_right_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Размещаем изображение на черном фоне
    new_left_image[top:top + current_height, left:left + current_width] = left_image
    new_right_image[top:top + current_height, left:left + current_width] = right_image
    
    return new_left_image, new_right_image
            
def depth_processing(image):
    ''' Создание карты глубины для изображения '''
    
    # Вычисление глубины
    with torch.no_grad():
        depth = model_depth.infer_image(image)
        
    # Нормализация глубины
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())

    return depth_normalized

def image3d_processing_method1(image, depth, height, width):
    ''' Функция создания стереопары на основе исходного изображения и карты глубины.
        Метод1: более быстрый, контуры более сглаженные, но может быть менее точным
    '''
    
    # Вычисление значения для параллакса
    parallax = depth * PARALLAX_SCALE
    
    # Сетка координат
    y, x = np.indices((height, width), dtype=np.float32)

    # Вычисление смещений
    shift_left  = np.clip(x - parallax, 0, width - 1)
    shift_right = np.clip(x + parallax, 0, width - 1)

    # Применение смещений с cv2.remap
    left_image  = cv2.remap(image, shift_left,  y, interpolation=INTERPOLATION_TYPE)
    right_image = cv2.remap(image, shift_right, y, interpolation=INTERPOLATION_TYPE)

    return left_image, right_image
    
def image3d_processing_method2(image, depth, height, width):
    ''' Функция создания стереопары на основе исходного изображения и карты глубины.
        Метод2: немного медленнее первого метода, но может быть точнее
    '''
    
    # Вычисление значения для параллакса
    parallax = depth * PARALLAX_SCALE
    
    # Округление параллакса и преобразование в int32
    shift = np.round(parallax).astype(np.int32)

    # Сетка координат
    y, x = np.indices((height, width), dtype=np.int32)

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
    left_image  = cv2.inpaint(left_image,  mask_left,  INPAINT_RADIUS, cv2.INPAINT_TELEA)
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

def get_total_frames():
    ''' Определение точного количества фреймов в видео.
        Сначала пробуется первый вариант, он быстрее, но срабатывает редко.
        Если не сработал первый вариант, пробуется второй, он долгий, но обычно отрабатывает хорошо
    '''
    
    cmd1 = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", video_file_path]
    cmd2 = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-count_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", video_file_path]
    
    try:
        result = subprocess.check_output(cmd1).splitlines()[0].decode().strip()
        print(f"Вариант1: {result}")
        if result != "N/A":
            return int(result)
    except Exception:
        pass

    try:
        result = subprocess.check_output(cmd2).splitlines()[0].decode().strip()
        print(f"Вариант2: {result}")
        if result != "N/A":
            return int(result)
    except Exception:
        pass
    
    # Если оба варианта не сработали, возвращаем None
    print("Ошибка, не удалось определить исходное количество фреймов.")
    
    return None

def extract_frames(start_frame, end_frame):
    ''' Извлечение фреймов и распределение их по чанкам исходя из chunk_size '''
    
    frames_to_process = end_frame - start_frame + 1
    extracted_frames = []

    with frame_counter.get_lock():
        start_counter = frame_counter.value
        frame_counter.value += frames_to_process

    for chunk_start in range(start_frame, end_frame + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, end_frame)
        extract_frames_path = os.path.join(frames_path, f"file_%06d.png")

        cmd = [
            "ffmpeg", "-hwaccel", "cuda", "-i", video_file_path,
            "-vf", f"select='between(n,{chunk_start},{chunk_end})'",
            "-vsync", "0", "-start_number", str(chunk_start), extract_frames_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        print(cmd)

        for i in range(chunk_end - chunk_start + 1):
            frame_number = chunk_start + i
            frame_path = extract_frames_path % frame_number
            extracted_frames.append(frame_path)
                
    return extracted_frames
    
def chunk_processing(extracted_frames):
    ''' Старт обработки каждого заполненного чанка '''
    
    for frame_path in extracted_frames:
    
        # Извлекаем имя изображения для последующего сохранения 3D изображения
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]
        
        # Загрузка изображения
        image = cv2.imread(frame_path)
        
        # Размеры изображения
        height, width = image.shape[:2]

        # Запуск depth_processing и получение карты глубины
        depth = depth_processing(image)

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
        output_image3d_path = os.path.join(images3d_path, f'{frame_name}.jpg')
        cv2.imwrite(output_image3d_path, image3d, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #cv2.imwrite(output_image3d_path, image3d) # Если PNG

        # Удаление исходного файла
        os.remove(frame_path)
        
    with threads_count.get_lock():
        threads_count.value = max(1, threads_count.value - 1) # Уменьшение счетчика после завершения текущего потока

def run_processing():
    ''' Глобальная функция старта обработки с учетом многопоточности '''
    
    # Получение количества фреймов в видеофайле
    total_frames = get_total_frames()
                        
    # Управление потоками
    if isinstance(total_frames, int):
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            for start_frame in range(0, total_frames, chunk_size):
                end_frame = min(start_frame + chunk_size - 1, total_frames - 1)
                extracted_frames = extract_frames(start_frame, end_frame)
                future = executor.submit(chunk_processing, extracted_frames)
                futures.append(future)
            
            # Ожидаем завершения задач
            for future in futures:
                future.result()
                
        print("ГОТОВО.")
    else:
        print("Сначала нужно определить значение total_frames.")


# ЗАПУСК ОБРАБОТКИ
run_processing()


# Выгружаем модель и очищаем кеш Cuda
del model_depth
torch.cuda.empty_cache()