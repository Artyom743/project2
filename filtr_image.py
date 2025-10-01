#Загрузка библиотек
from skimage import data
import matplotlib.pyplot as plt
import numpy as np

#Загружаем изображение
def init_image():
    """
    Загружает тестовое изображение астронавта из skimage.
    
    Returns:
        numpy.ndarray: Загруженное изображение в формате массива numpy
        
    Raises:
        ValueError: Если изображение не удалось загрузить
    """
    try:
        astronaut_image = data.astronaut()
        if astronaut_image is None:
            raise ValueError("Не удалось загрузить изображение")
        return astronaut_image
    except Exception as e:
        raise ValueError(f"Ошибка загрузки изображения: {e}")

#Переводим во float
def to_float(astronaut_image):
    """
    Преобразует изображение в формат float64 для математических операций.
    
    Args:
        astronaut_image (numpy.ndarray): Входное изображение
        
    Returns:
        numpy.ndarray: Изображение в формате float64
        
    Raises:
        TypeError: Если входные данные не являются numpy массивом
        ValueError: Если изображение пустое
    """
    try:
        if not isinstance(astronaut_image, np.ndarray):
            raise TypeError("Изображение должно быть numpy массивом")
        if astronaut_image.size == 0:
            raise ValueError("Изображение пустое")
        return astronaut_image.astype(np.float64)
    except Exception as e:
        raise ValueError(f"Ошибка преобразования в float: {e}")

#Проверяем матрицу сепии
def validate_sepia_matrix(sepia_matrix):
    """
    Проверяет корректность матрицы преобразования в сепию.
    
    Args:
        sepia_matrix (numpy.ndarray or list): Матрица 3x3 для преобразования
        
    Returns:
        numpy.ndarray: Проверенная матрица сепии
        
    Raises:
        TypeError: Если матрица не является numpy массивом или списком
        ValueError: Если матрица имеет неверный размер или содержит некорректные значения
    """
    try:
        if not isinstance(sepia_matrix, (np.ndarray, list)):
            raise TypeError("Матрица сепии должна быть numpy массивом или списком")
        if sepia_matrix.shape != (3, 3):
            raise ValueError("Матрица сепии должна быть размером 3x3")
        if np.any(np.isnan(sepia_matrix)) or np.any(np.isinf(sepia_matrix)):
            raise ValueError("Матрица сепии содержит некорректные значения")
        return sepia_matrix
    except Exception as e:
        raise ValueError(f"Ошибка валидации матрицы: {e}")

#Применяем матрицу сепии к изображению и ограничиваем 0-255
def sepia_to_image(sepia_matrix, float64_image, height, width, channels):
    """
    Применяет сепия-фильтр к изображению с помощью матричного преобразования.
    
    Args:
        sepia_matrix (numpy.ndarray): Матрица преобразования 3x3
        float64_image (numpy.ndarray): Изображение в формате float64
        height (int): Высота изображения
        width (int): Ширина изображения
        channels (int): Количество каналов изображения
        
    Returns:
        numpy.ndarray: Обработанное изображение в формате uint8
        
    Raises:
        ValueError: Если изображение не инициализировано или имеет неверное количество каналов
    """
    try:
        if float64_image is None:
            raise ValueError("Изображение не инициализировано")
        if channels != 3:
            raise ValueError("Изображение должно иметь 3 канала RGB")
        result = np.zeros_like(float64_image)
        for i in range(height):
            for j in range(width):
                result[i, j] = np.dot(sepia_matrix, float64_image[i,j])
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    except Exception as e:
        raise ValueError(f"Ошибка применения фильтра: {e}")

#Вывод до и после
def print_images(astronaut_image, result):
    """
    Отображает оригинальное и обработанное изображения рядом для сравнения.
    
    Args:
        astronaut_image (numpy.ndarray): Оригинальное изображение
        result (numpy.ndarray): Изображение после применения сепия-фильтра
        
    Raises:
        ValueError: Если одно из изображений не инициализировано
    """
    try:
        if astronaut_image is None or result is None:
            raise ValueError("Изображения для отображения не инициализированы")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(astronaut_image)
        ax1.set_title('Оригинальное изображение')
        ax1.axis('off')
        ax2.imshow(result)
        ax2.set_title('Сепия фильтр')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        raise ValueError(f"Ошибка отображения изображений: {e}")

#Основная функция
def main():
    """
    Основная функция программы.

    """
    try:
        astronaut_image = init_image()
        float64_image = to_float(astronaut_image)
        sepia_matrix = np.array([
            [0.547, 0.857, 0.264],
            [0.186, 0.479, 0.498],
            [0.979, 0.612, 0.492]
        ])
        sepia_matrix = validate_sepia_matrix(sepia_matrix)
        height, width, channel = float64_image.shape
        result = sepia_to_image(sepia_matrix, float64_image, height, width, channel)
        print_images(astronaut_image, result)
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()