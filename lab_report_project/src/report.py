import os
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np

def save_scatter(
    samples: List[np.ndarray], 
    output_dir: Union[str, Path], 
    title: str,
    xlabel: str = "X",
    ylabel: str = "Y"
) -> str:
    """
    Создает и сохраняет scatter plot для набора выборок.
    
    Аргументы:
        samples: Список массивов с выборками
        output_dir: Директория для сохранения графика
        title: Заголовок графика
        xlabel: Подпись оси X
        ylabel: Подпись оси Y
        
    Возвращает:
        Путь к сохраненному файлу с графиком
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Создаем scatter plot для каждой выборки
    for i, sample in enumerate(samples):
        if len(sample) == 0:
            continue
        plt.scatter(
            sample[:, 0], 
            sample[:, 1], 
            label=f'Класс {i+1}', 
            alpha=0.6,
            s=50,
            edgecolors='w',
            linewidth=0.5
        )
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Улучшаем читаемость осей
    plt.tight_layout()
    
    # Сохраняем график
    filename = f"{title.lower().replace(' ', '_')}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(filepath)

def generate_report(
    estimations: List[Tuple[np.ndarray, np.ndarray]],
    distances: Dict[str, float],
    data_files: List[str],
    img_paths: List[str],
    output_path: Union[str, Path]
) -> None:
    """
    Генерирует Markdown отчет с результатами анализа.
    
    Аргументы:
        estimations: Список кортежей (мат. ожидание, ковариационная матрица)
        distances: Словарь с расстояниями между распределениями
        data_files: Список путей к файлам с данными
        img_paths: Список путей к графикам
        output_path: Путь для сохранения отчета
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Заголовок отчета
        f.write("# Лабораторная работа: Моделирование случайных векторов\n\n")
        
        # Раздел с параметрами распределений
        f.write("## Оценки параметров распределений\n\n")
        for i, (mean, cov) in enumerate(estimations):
            f.write(f"### Класс {i+1}:\n")
            f.write("**Математическое ожидание:**\n")
            f.write(f"```\n{np.array2string(mean, precision=4, suppress_small=True)}\n```\n\n")
            f.write("**Ковариационная матрица:**\n")
            f.write(f"```\n{np.array2string(cov, precision=4, suppress_small=True)}\n```\n\n")
        
        # Раздел с расстояниями
        f.write("## Расстояния между распределениями\n\n")
        for metric, value in distances.items():
            f.write(f"- **{metric}:** {value:.4f}\n")
        f.write("\n")
        
        # Раздел с визуализациями
        f.write("## Визуализация распределений\n\n")
        for img_path in img_paths:
            img_name = Path(img_path).stem.replace('_', ' ').title()
            f.write(f"### {img_name}\n")
            f.write(f"![]({img_path})\n\n")
        
        # Раздел с файлами данных
        f.write("## Файлы с данными\n\n")
        for file_path in data_files:
            f.write(f"- `{file_path}`\n")
        
        # Заключение
        f.write("\n## Выводы\n\n")
        f.write("1. Были успешно сгенерированы выборки случайных векторов с заданными параметрами.\n")
        f.write("2. Проведена оценка параметров распределений по выборкам.\n")
        f.write("3. Рассчитаны расстояния между распределениями.\n")
        f.write("4. Построены визуализации распределений.\n")
        f.write("5. Все результаты сохранены в соответствующие файлы.\n")

    print(f"Отчет успешно сохранен в {output_path}")
