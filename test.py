import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# 设置随机数种子
np.random.seed(42)

# 1. 生成P和Q的样本点 (二维正态分布)
n_samples = 1000000  # 采样点数量
P_samples = np.random.normal(0, 1, (n_samples, 2))  # P ~ N((0, 0), I)
Q_samples = np.random.normal(1, 1, (n_samples, 2))  # Q ~ N((1, 1), I)

# 2. 计算 Wasserstein-1 距离 (真实的 Wasserstein-1 距离)
# 由于这是两个二维正态分布，可以通过公式计算它们之间的 Wasserstein 距离
# 对于二维正态分布，Wasserstein-1 距离的公式为
# W1(P, Q) = ||mu_P - mu_Q||_2 = sqrt((mu_P[0] - mu_Q[0])^2 + (mu_P[1] - mu_Q[1])^2)
W1_real = np.linalg.norm(np.mean(P_samples, axis=0) - np.mean(Q_samples, axis=0))
print(f"真实的 Wasserstein-1 距离: {W1_real}")

# 3. 计算 Sliced Wasserstein 距离
def sliced_wasserstein_distance(P, Q, n_projections=100):
    n_samples = P.shape[0]
    dim = P.shape[1]
    
    # 随机生成投影方向
    projections = np.random.randn(n_projections, dim)
    projections /= np.linalg.norm(projections, axis=1)[:, None]  # 标准化为单位向量
    
    # 存储每个投影方向的 Wasserstein-1 距离
    swd = 0
    
    for proj in projections:
        # 投影样本点
        p_proj = P.dot(proj)
        q_proj = Q.dot(proj)
        
        # 排序投影值
        p_proj_sorted = np.sort(p_proj)
        q_proj_sorted = np.sort(q_proj)
        
        # 计算一维 Wasserstein-1 距离
        swd += np.mean(np.abs(p_proj_sorted - q_proj_sorted))
    
    # 取平均值
    return swd / n_projections

# 计算 Sliced Wasserstein 距离
swd_estimate = sliced_wasserstein_distance(P_samples, Q_samples)
print(f"估计的 Sliced Wasserstein 距离: {swd_estimate}")

# 4. 计算误差
error = np.abs(W1_real - swd_estimate)
print(f"Wasserstein 距离和 Sliced Wasserstein 距离之间的误差: {error}")
