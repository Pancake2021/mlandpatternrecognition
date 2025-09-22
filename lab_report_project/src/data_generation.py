import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Union

def generate_normal_samples(
    means: List[np.ndarray],
    covs: List[np.ndarray],
    n_samples: int,
    output_dir: Union[str, Path]
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Генерирует выборки из многомерного нормального распределения.
    
    Аргументы:
        means: Список векторов математических ожиданий
        covs: Список ковариационных матриц
        n_samples: Количество сэмплов для каждой выборки
        output_dir: Директория для сохранения сгенерированных данных
        
    Возвращает:
        Кортеж (список массивов с выборками, список путей к сохранённым файлам)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    file_paths = []
    
    for i, (mean, cov) in enumerate(zip(means, covs)):
        # Генерация выборки
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
