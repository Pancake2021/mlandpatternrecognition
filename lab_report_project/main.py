#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Основной скрипт для выполнения лабораторной работы по моделированию случайных векторов.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Импортируем наши модули
from src.data_generation import generate_normal_samples, generate_binary_samples
from src.analysis import estimate_parameters, mahalanobis_dist, bhattacharyya_dist
from src.report import save_scatter, generate_report

def setup_directories() -> Dict[str, Path]:
    """Создает необходимые директории и возвращает пути к ним."""
    base_dir = Path(__file__).parent
    dirs = {
        'base': base_dir,
        'data': base_dir / 'data' / 'generated',
        'reports': base_dir / 'reports',
        'plots': base_dir / 'reports' / 'plots'
    }
    
    # Создаем директории, если они не существуют
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def main():
    """Основная функция для выполнения лабораторной работы."""
    # Настройка путей
    dirs = setup_directories()
    
    # Параметры для генерации данных
    N = 500  # Количество сэмплов в каждой выборке
    
    # Параметры нормальных распределений
    means = [
        np.array([1, 2]),
        np.array([3, 0]),
        np.array([2, 3])
    ]
    
    # Ковариационные матрицы (равные и неравные)
    covs_equal = [
        np.array([[1, 0.5], [0.5, 1]]),
        np.array([[1, 0.5], [0.5, 1]])
    ]
    
    covs_unequal = [
        np.array([[1, 0.2], [0.2, 1]]),
        np.array([[1, 0.8], [0.8, 1]]),
        np.array([[1, 0], [0, 1]])
    ]
    
    # 1. Генерация нормально распределенных выборок
    print("Генерация нормально распределенных выборок...")
    samples_eq, files_eq = generate_normal_samples(
        means[:2], covs_equal, N, dirs['data']
    )
    
    samples_uneq, files_uneq = generate_normal_samples(
        means, covs_unequal, N, dirs['data']
    )
    
    # 2. Генерация бинарных выборок
    print("Генерация бинарных выборок...")
    binary_samples, binary_files = generate_binary_samples(
        N, 0.3, 2, dirs['data']
    )
    
    # 3. Оценка параметров распределений
    print("Оценка параметров распределений...")
    estimations = estimate_parameters(samples_uneq)
    
    # 4. Расчет расстояний между распределениями
    print("Расчет расстояний между распределениями...")
    dist_mahalanobis = mahalanobis_dist(
        estimations[0][0], 
        estimations[1][0], 
        (estimations[0][1] + estimations[1][1]) / 2
    )
    
    dist_bhattacharyya = bhattacharyya_dist(
        estimations[0][0], 
        estimations[1][0], 
        estimations[0][1], 
        estimations[1][1]
    )
    
    distances = {
        "Расстояние Махаланобиса": dist_mahalanobis,
        "Расстояние Бхатачария": dist_bhattacharyya
    }
    
    # 5. Визуализация результатов
    print("Создание визуализаций...")
    img_eq = save_scatter(
        samples_eq, 
        dirs['plots'], 
        "Распределения с равными ковариационными матрицами",
        "X",
        "Y"
    )
    
    img_uneq = save_scatter(
        samples_uneq, 
        dirs['plots'], 
        "Распределения с разными ковариационными матрицами",
        "X",
        "Y"
    )
    
    # 6. Генерация отчета
    print("Формирование отчета...")
    report_path = dirs['reports'] / 'lab_report.md'
    generate_report(
        estimations=estimations,
        distances=distances,
        data_files=files_uneq + files_eq + binary_files,
        img_paths=[img_eq, img_uneq],
        output_path=report_path
    )
    
    print(f"\nЛабораторная работа успешно выполнена!")
    print(f"Отчет сохранен в: {report_path}")

if __name__ == "__main__":
    main()
