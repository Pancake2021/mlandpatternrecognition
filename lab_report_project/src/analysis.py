from typing import List, Tuple
import numpy as np
from scipy.spatial import distance

def estimate_parameters(samples: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Оценивает параметры распределения для набора выборок.
    
    Аргументы:
        samples: Список массивов с выборками
        
    Возвращает:
        Список кортежей (оценка мат. ожидания, оценка ковариационной матрицы)
    """
    estimates = []
    for sample in samples:
        mean_est = np.mean(sample, axis=0)
        cov_est = np.cov(sample, rowvar=False)
        estimates.append((mean_est, cov_est))
    return estimates

def mahalanobis_dist(
    mean1: np.ndarray, 
    mean2: np.ndarray, 
    cov: np.ndarray
) -> float:
    """
    Вычисляет расстояние Махаланобиса между двумя векторами.
    
    Формула: D² = (x - y)ᵀ · Σ⁻¹ · (x - y)
    
    Аргументы:
        mean1: Первый вектор средних (x)
        mean2: Второй вектор средних (y)
        cov: Ковариационная матрица (Σ)
        
    Возвращает:
        Расстояние Махаланобиса между векторами
    """
    # Вычисляем разность векторов
    diff = mean1 - mean2
    
    # Вычисляем обратную ковариационную матрицу
    inv_cov = np.linalg.inv(cov)
    
    # Вычисляем расстояние Махаланобиса
    # (x-y)ᵀ · Σ⁻¹ · (x-y)
    distance_squared = np.dot(np.dot(diff.T, inv_cov), diff)
    
    # Возвращаем квадратный корень (если нужен сам D, а не D²)
    return np.sqrt(distance_squared)

def bhattacharyya_dist(
    mean1: np.ndarray, 
    mean2: np.ndarray, 
    cov1: np.ndarray, 
    cov2: np.ndarray
) -> float:
    """
    Вычисляет расстояние Бхатачария между двумя многомерными нормальными распределениями.
    
    Аргументы:
        mean1: Вектор средних первого распределения
        mean2: Вектор средних второго распределения
        cov1: Ковариационная матрица первого распределения
        cov2: Ковариационная матрица второго распределения
        
    Возвращает:
        Расстояние Бхатачария между распределениями
    """
    cov_avg = (cov1 + cov2) / 2
    mean_diff = mean1 - mean2
    
    # Первое слагаемое: взвешенное расстояние между средними
    inv_cov_avg = np.linalg.inv(cov_avg)
    term1 = 0.125 * np.dot(np.dot(mean_diff, inv_cov_avg), mean_diff)
    
    # Второе слагаемое: расхождение ковариаций
    det_cov_avg = np.linalg.det(cov_avg)
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)
    
    # Избегаем деления на ноль и логарифма от нуля
    if det_cov1 <= 0 or det_cov2 <= 0 or det_cov_avg <= 0:
        return float('inf')
        
    term2 = 0.5 * np.log(det_cov_avg / np.sqrt(det_cov1 * det_cov2))
    
    return term1 + term2
