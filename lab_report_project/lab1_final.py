import numpy as np
import matplotlib.pyplot as plt
import os


def generate_normal_vector(mean_vector, covariance_matrix, N, save_filename):
    """
    Generates N realizations of n-dimensional normal vector.
    Parameters:
        mean_vector: mean vector (n, 1)
        covariance_matrix: covariance matrix (n, n)
        N: sample size
        save_filename: filename for saving
    """
    uniform_vector = np.array([np.zeros(N), np.zeros(N)])

    # генерим равномерные выборки и усредняем по цпт
    cpt_len = 50
    for i in range(cpt_len):
        uni = np.array([np.random.uniform(0, 6, N),
                       np.random.uniform(0, 6, N)])
        uniform_vector += uni
    uniform_vector /= cpt_len

    m = [[3], [3]]
    sigma = [[np.sqrt(3)], [np.sqrt(3)]]

    standart_vector = np.sqrt(
        cpt_len) * (uniform_vector - m) / sigma  # ЦПТ в форме Леви

    # Find transformation matrix A (lower triangular) for 2D case
    A = np.zeros((2, 2))
    A[0, 0] = np.sqrt(covariance_matrix[0, 0])
    A[1, 0] = covariance_matrix[0, 1] / A[0, 0]
    A[1, 1] = np.sqrt(covariance_matrix[1, 1] - A[1, 0] ** 2)

    # Transformation: X = A * ξ + M
    # A.shape = (2,2), xi.shape = (2, N), result.shape = (2, N)
    x = A @ standart_vector + mean_vector

    np.save(save_filename, x)
    return x


def estimate_mean(data_file):
    """Estimates mean vector from data file"""
    x = np.load(data_file)
    m_estimate = np.mean(x, axis=1)
    # Return as column vector for consistency with other functions
    return m_estimate.reshape(-1, 1)


def estimate_covariance(data_file):
    """Estimates covariance matrix from data file"""
    x = np.load(data_file)  # x.shape = (2, N)

    m_estimate = estimate_mean(data_file)  # shape (2, 1)
    x_centered = x - m_estimate  # Thanks to NumPy broadcasting
    # Â = (1/N) * Σ (x_centered_i) * (x_centered_i)^T = (1/N) * (X_centered @ X_centered.T)
    B_estimate = (x_centered @ x_centered.T) / x.shape[1]

    return B_estimate


def bhattacharyya_distance(M1, M2, B1, B2):
    """Calculates Bhattacharyya distance between two normal distributions"""
    M1 = M1.reshape(-1, 1)
    M2 = M2.reshape(-1, 1)

    half_sum_cov = (B1 + B2) / 2
    diff_mean = M1 - M2

    term1 = 0.125 * diff_mean.T @ np.linalg.inv(half_sum_cov) @ diff_mean
    det_half = np.linalg.det(half_sum_cov)
    det_prod = np.linalg.det(B1) * np.linalg.det(B2)
    term2 = 0.5 * np.log(det_half / np.sqrt(det_prod))

    rho_b = term1 + term2
    return rho_b.item()  # Extract scalar from 1x1 matrix


def mahalanobis_distance(M1, M2, B):
    """Calculates Mahalanobis distance between two vectors relative to covariance matrix B"""
    M1 = M1.reshape(-1, 1)
    M2 = M2.reshape(-1, 1)
    diff_mean = M1 - M2
    rho_m = diff_mean.T @ np.linalg.inv(B) @ diff_mean
    return rho_m.item()  # Square root for true Mahalanobis distance


def plot_normal_data(data_list, labels, colors, title, filename):
    """
    Visualizes normally distributed data.
    """
    plt.figure(figsize=(10, 8))
    for data, label, color in zip(data_list, labels, colors):
        plt.scatter(data[0, :], data[1, :], alpha=0.7,
                    label=label, color=color, s=20)

    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # ==================== PARAMETER SETUP ====================
    N_SAMPLES = 200

    # Task 2: Two distributions with EQUAL covariance matrices
    M1 = np.array([[-1.0], [1.0]])  # Center of first distribution
    M2 = np.array([[0.0], [1.0]])  # Center of second distribution
    B_EQUAL = np.array([[0.3, 0.35], [0.1, 0.65]])

    # Task 3: Three distributions with DIFFERENT covariance matrices
    M3 = np.array([[-1.0], [-1.0]])  # Center of third distribution
    B1 = np.array([[0.2, -0.35], [-0.15, 0.65]])
    B2 = np.array([[0.25, 0.3], [0.15, 0.45]])
    B3 = np.array([[0.25, -0.3], [-0.15, 0.45]])

    task2_vector1_path = "task2_vector1.npy"
    task2_vector2_path = "task2_vector2.npy"
    task3_vector1_path = "task3_vector1.npy"
    task3_vector2_path = "task3_vector2.npy"
    task3_vector3_path = "task3_vector3.npy"
    # ==============================================================

    print("=" * 60)
    print("LAB WORK #1: DATA MODELING FOR PATTERN RECOGNITION")
    print("=" * 60)

    # ==================== TASK 2 ====================
    print("\n--- TASK 2: Two normal distributions with equal matrices ---")

    # Generate data
    data1_task2 = generate_normal_vector(
        M1, B_EQUAL, N_SAMPLES, task2_vector1_path)
    data2_task2 = generate_normal_vector(
        M2, B_EQUAL, N_SAMPLES, task2_vector2_path)

    # Parameter estimation (pass filenames, not data)
    m1_est = estimate_mean(task2_vector1_path)
    m2_est = estimate_mean(task2_vector2_path)
    b1_est = estimate_covariance(task2_vector1_path)
    b2_est = estimate_covariance(task2_vector2_path)

    print("Estimates for distribution 1:")
    print(f"M1: {m1_est.flatten()}")
    print(f"B1:\n{b1_est}")

    print("\nEstimates for distribution 2:")
    print(f"M2: {m2_est.flatten()}")
    print(f"B2:\n{b2_est}")

    # Distance calculation
    dist_m = mahalanobis_distance(m1_est, m2_est, B_EQUAL)
    print(f"\nMahalanobis distance: {dist_m:.4f}")

    # Visualization
    plot_normal_data([data1_task2, data2_task2],
                     ['Distribution 1', 'Distribution 2'],
                     ['red', 'blue'],
                     'Task 2: Two normal distributions (equal covariance matrices)',
                     'task2_plot.png')

    # ==================== TASK 3 ====================
    print("\n--- TASK 3: Three normal distributions with different matrices ---")

    # Generate data
    data1_task3 = generate_normal_vector(M1, B1, N_SAMPLES, task3_vector1_path)
    data2_task3 = generate_normal_vector(M2, B2, N_SAMPLES, task3_vector2_path)
    data3_task3 = generate_normal_vector(M3, B3, N_SAMPLES, task3_vector3_path)

    # Parameter estimation
    m1_est3 = estimate_mean(task3_vector1_path)
    m2_est3 = estimate_mean(task3_vector2_path)
    m3_est3 = estimate_mean(task3_vector3_path)
    b1_est3 = estimate_covariance(task3_vector1_path)
    b2_est3 = estimate_covariance(task3_vector2_path)
    b3_est3 = estimate_covariance(task3_vector3_path)

    # Calculate Bhattacharyya distances
    dist_b_12 = bhattacharyya_distance(m1_est3, m2_est3, b1_est3, b2_est3)
    dist_b_23 = bhattacharyya_distance(m2_est3, m3_est3, b2_est3, b3_est3)
    dist_b_13 = bhattacharyya_distance(m1_est3, m3_est3, b1_est3, b3_est3)

    print("Bhattacharyya distances:")
    print(f"Between distributions 1 and 2: {dist_b_12:.4f}")
    print(f"Between distributions 2 and 3: {dist_b_23:.4f}")
    print(f"Between distributions 1 and 3: {dist_b_13:.4f}")

    # Visualization
    plot_normal_data([data1_task3, data2_task3, data3_task3],
                     ['Distribution 1', 'Distribution 2', 'Distribution 3'],
                     ['red', 'blue', 'green'],
                     'Task 3: Three normal distributions (different covariance matrices)',
                     'task3_plot.png')

    print("\nWork completed successfully!")
