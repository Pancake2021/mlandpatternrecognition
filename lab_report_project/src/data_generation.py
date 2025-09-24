import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Union, Optional


def generate_normal_clt(mean: float = 0.0, std: float = 1.0, size: int = 1, n_uniform: int = 12) -> np.ndarray:
    """
    Генерирует нормально распределенные числа с использованием ЦПТ.
    
    Аргументы:
        mean: Математическое ожидание
        std: Среднеквадратическое отклонение
        size: Количество сэмплов
        n_uniform: Количество равномерных случайных величин для суммирования (по умолчанию 12)
        
    Возвращает:
        Массив нормально распределенных чисел
    """
    # Генерируем n_uniform равномерных случайных величин на интервале [0, 1]
    uniform_samples = np.random.uniform(0, 1, size=(n_uniform, size))
    
    # (частный случай ЦПТ)
    # Сумма 12 U[0,1] имеет мат. ожидание 6 и дисперсию 1
    z = (np.sum(uniform_samples, axis=0) - n_uniform/2) / np.sqrt(n_uniform/12)
    
    # Масштабируем к нужным параметрам
    return mean + std * z


def generate_multivariate_normal_clt(mean: np.ndarray, cov: np.ndarray, n_samples: int = 1) -> np.ndarray:
    """
    Генерирует многомерное нормальное распределение с использованием ЦПТ.
    
    Аргументы:
        mean: Вектор средних значений
        cov: Ковариационная матрица
        n_samples: Количество сэмплов
        
    Возвращает:
        Матрицу размера (n_samples, n_features) с нормально распределенными векторами
    """
    n_features = len(mean)
    
    # Генерируем стандартные нормальные величины с помощью ЦПТ
    z = np.zeros((n_samples, n_features))
    for i in range(n_features):
        z[:, i] = generate_normal_clt(mean=0, std=1, size=n_samples)
    
    # Разложение Холецкого для ковариационной матрицы
    L = np.linalg.cholesky(cov)
    
    # Преобразуем к нужному распределению
    samples = mean + np.dot(z, L.T)
    
    return samples

def generate_normal_samples(
    means: List[np.ndarray],
    covs: List[np.ndarray],
    n_samples: int,
    output_dir: Union[str, Path],
    use_clt: bool = True
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Генерирует выборки из многомерного нормального распределения.
    
    Аргументы:
        means: Список векторов математических ожиданий
        covs: Список ковариационных матриц
        n_samples: Количество сэмплов для каждой выборки
        output_dir: Директория для сохранения сгенерированных данных
        use_clt: Если True, использует ЦПТ для генерации нормальных величин
        
    Возвращает:
        Кортеж (список массивов с выборками, список путей к сохранённым файлам)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    file_paths = []
    
    for i, (mean, cov) in enumerate(zip(means, covs)):
        if use_clt:
            # Используем нашу реализацию с ЦПТ
            sample = generate_multivariate_normal_clt(mean, cov, n_samples)
        else:
            # Используем встроенную функцию для сравнения
            sample = np.random.multivariate_normal(mean, cov, n_samples)
        
        samples.append(sample)
        
        # Сохранение в файл
        file_path = output_dir / f"normal_sample_{i+1}.npy"
        np.save(file_path, sample)
        file_paths.append(str(file_path))
    
    return samples, file_paths

def generate_binary_samples(
    n_samples: int,
    probability: float,
    n_vectors: int,
    output_dir: Union[str, Path]
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Генерирует бинарные случайные векторы.
    
    Аргументы:
        n_samples: Количество сэмплов в каждой выборке
        probability: Вероятность успеха (1)
        n_vectors: Количество генерируемых векторов
        output_dir: Директория для сохранения сгенерированных данных
        
    Возвращает:
        Кортеж (список массивов с выборками, список путей к сохранённым файлам)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    file_paths = []
    
    for i in range(n_vectors):
        # Генерация бинарной выборки
        sample = np.random.binomial(1, probability, (n_samples, 2))
        samples.append(sample)
        
        # Сохранение в файл
        file_path = output_dir / f"binary_sample_{i+1}.npy"
        np.save(file_path, sample)
        file_paths.append(str(file_path))
    
    return samples, file_paths
