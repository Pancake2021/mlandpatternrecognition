#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Лабораторная работа 1.2.2
Моделирование случайных векторов
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

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
    """
    Основная функция для выполнения лабораторной работы 1.2.2.
    Выполняет моделирование случайных векторов с нормальным и биномиальным распределениями.
    """
    # Настройка путей
    dirs = setup_directories()
    
    # 1. Параметры для генерации данных
    print("1. Настройка параметров генерации...")
    N = 200  # Количество сэмплов в каждой выборке
    
    # Параметры нормальных распределений
    means = [
        np.array([1, 0]),
        np.array([-1, 1]),
        np.array([1, -2])
    ]
    
    # Общая ковариационная матрица для первых двух распределений
    common_cov = np.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    covs_equal = [common_cov, common_cov]
    
    # 3. Параметры для трех нормальных распределений с разными ковариационными матрицами
    means_3d = [
        np.array([0, 0]),    # Третье распределение
        np.array([4, 1]),    # Четвертое распределение
        np.array([-3, 2])    # Пятое распределение
    ]
    
    covs_unequal = [
        np.array([[1.0, 0.3], [0.3, 0.8]]),  # Для третьего распределения
        np.array([[0.6, 0.1], [0.1, 1.2]]),  # Для четвертого распределения
        np.array([[1.5, -0.4], [-0.4, 0.7]])  # Для пятого распределения
    ]
    
    # 4. Генерация выборок с равными ковариационными матрицами
    print("\n2. Генерация выборок с равными ковариационными матрицами...")
    samples_eq, files_eq = generate_normal_samples(
        means_2d, covs_equal, N, dirs['data']
    )
    
    # 5. Генерация выборок с разными ковариационными матрицами
    print("3. Генерация выборок с разными ковариационными матрицами...")
    samples_uneq, files_uneq = generate_normal_samples(
        means_3d, covs_unequal, N, dirs['data']
    )
    
    # 6. Оценка параметров распределений
    print("\n4. Оценка параметров распределений...")
    estimations = estimate_parameters(samples_uneq)
    
    # Вывод оценок параметров
    print("\nОценки параметров распределений:")
    for i, (mean_est, cov_est) in enumerate(estimations, 1):
        print(f"\nРаспределение {i}:")
        print(f"Оценка мат. ожидания:\n{mean_est}")
        print(f"Оценка ковариационной матрицы:\n{cov_est}")
    
    # 7. Расчет расстояний между распределениями
    print("\n5. Расчет расстояний между распределениями...")
    
    # Расстояние между первыми двумя распределениями
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
    
    print("\nРасстояния между распределениями:")
    for name, value in distances.items():
        print(f"{name}: {value:.4f}")
    
    # 8. Генерация бинарных выборок с вероятностью 0.3
    print("\n6. Генерация бинарных выборок...")
    binary_samples, binary_files = generate_binary_samples(
        N, 0.3, 2, dirs['data']
    )
    
    # 9. Визуализация результатов
    print("\n7. Создание визуализаций...")
    
    # Визуализация распределений с равными ковариационными матрицами
    img_eq = save_scatter(
        samples_eq, 
        dirs['plots'], 
        "Два нормальных распределения с равными ковариационными матрицами",
        "X",
        "Y",
        colors=['blue', 'red']
    )
    
    # Визуализация распределений с разными ковариационными матрицами
    img_uneq = save_scatter(
        samples_uneq, 
        dirs['plots'], 
        "Три нормальных распределения с разными ковариационными матрицами",
        "X",
        "Y",
        colors=['green', 'purple', 'orange']
    )
    
    # 10. Генерация отчета
    print("\n8. Формирование отчета...")
    report_path = dirs['reports'] / 'lab_report_1.2.2.md'
    generate_report(
        means=means_2d + means_3d,
        covs=covs_equal + covs_unequal,
        estimations=estimations,
        distances=distances,
        data_files=files_uneq + files_eq + binary_files,
        img_paths=[img_eq, img_uneq],
        output_path=report_path
    )
    
    print("\n" + "="*60)
    print("Лабораторная работа 1.2.2 успешно выполнена!")
    print(f"Отчет сохранен в: {report_path}")
    print("="*60)

if __name__ == "__main__":
    main()
