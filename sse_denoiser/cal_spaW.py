# # 计算空间权重矩阵
# import os
# import joblib
# import numpy as np
# from geopy.distance import geodesic

# def calculate_and_save_weights(data_path, output_path, alpha=0.05, d_threshold=300.0):
#     print(f"Loading dataset from: {data_path}")
#     if not os.path.exists(data_path):
#         # Fallback for demonstration/if user runs it in a different env
#         print(f"Warning: {data_path} not found. Please ensure the path is correct.")
#         return

#     data_dict = joblib.load(data_path)
#     # Extract station coordinates (assumed shape [200, 3] or [200, 2])
#     coords = data_dict['station_coordinates']
#     # Use only Lat, Lon
#     coords = coords[:, :2]
#     N = coords.shape[0]

#     dist_matrix = np.zeros((N, N))
#     print(f"Calculating geodesic distances for {N} stations...")

#     for i in range(N):
#         for j in range(i + 1, N):
#             d = geodesic((coords[i, 0], coords[i, 1]), (coords[j, 0], coords[j, 1])).km
#             dist_matrix[i, j] = d
#             dist_matrix[j, i] = d

#     print(f"Applying soft-margin decay (alpha={alpha}, threshold={d_threshold})...")
#     W = 1.0 / (1.0 + np.exp(-alpha * (dist_matrix - d_threshold)))
#     np.fill_diagonal(W, 0.0)

#     # Save to joblib
#     save_data = {
#         'weight_matrix': W,
#         'distance_matrix': dist_matrix,
#         'alpha': alpha,
#         'd_threshold': d_threshold,
#         'station_coordinates': coords
#     }
#     joblib.dump(save_data, output_path)
#     print(f"Success: Weights saved to {output_path}")

# if __name__ == '__main__':
#     # Configuration - User can adjust these paths
#     DATASET_PATH = r'D:\see-dn\data\small_dataset_10000.data'
#     OUTPUT_FILE = 'spatial_weights.joblib'

#     calculate_and_save_weights(DATASET_PATH, OUTPUT_FILE)


# 可视化结果---------------------------------------------------------------------------
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_spatial_weights(file_path, station_idx=0):
    # 1. 加载数据
    print(f"正在加载结果文件: {file_path}")
    data = joblib.load(file_path)
    
    W = data['weight_matrix']
    dist_matrix = data['distance_matrix']
    coords = data['station_coordinates']
    alpha = data['alpha']
    d_threshold = data['d_threshold']
    
    # 设置中文字体（如果是中文环境）
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    # 创建一个画布，包含三个子图
    fig = plt.figure(figsize=(18, 5))

    # --- 图 1: 权重矩阵热力图 ---
    ax1 = fig.add_subplot(131)
    sns.heatmap(W[:50, :50], cmap='viridis', ax=ax1) # 仅展示前50个站点防止过挤
    ax1.set_title(f"权重矩阵热力图 (局部 50x50)\nalpha={alpha}, threshold={d_threshold}")
    ax1.set_xlabel("站点索引")
    ax1.set_ylabel("站点索引")

    # --- 图 2: 权重 vs 距离 的曲线图 ---
    ax2 = fig.add_subplot(132)
    # 取矩阵的上三角部分（排除对角线）来画散点
    flat_dist = dist_matrix.flatten()
    flat_W = W.flatten()
    # 采样一部分点进行绘图以免卡顿
    sample_idx = np.random.choice(len(flat_dist), min(2000, len(flat_dist)), replace=False)
    
    ax2.scatter(flat_dist[sample_idx], flat_W[sample_idx], alpha=0.5, s=10, label='计算采样点')
    ax2.axvline(x=d_threshold, color='r', linestyle='--', label='距离阈值')
    ax2.set_title("权重与距离的衰减关系")
    ax2.set_xlabel("距离 (km)")
    ax2.set_ylabel("权重值 W")
    ax2.legend()

    # --- 图 3: 空间位置与特定站点的连接图 ---
    ax3 = fig.add_subplot(133)
    # 绘制所有站点
    ax3.scatter(coords[:, 1], coords[:, 0], c='gray', s=10, alpha=0.3, label='所有站点')
    
    # 选取一个特定站点（默认索引0）作为参考
    ref_coords = coords[station_idx]
    ax3.scatter(ref_coords[1], ref_coords[0], c='red', s=50, marker='*', label=f'参考站点 {station_idx}')
    
    # 获取该站点对其他所有站点的权重
    weights_from_ref = W[station_idx]
    
    # 绘制热点，颜色越深表示权重越高
    sc = ax3.scatter(coords[:, 1], coords[:, 0], c=weights_from_ref, 
                    cmap='YlOrRd', s=30, alpha=0.6, edgecolors='none')
    plt.colorbar(sc, ax=ax3, label='权重大小')
    
    ax3.set_title(f"以站点 {station_idx} 为中心的权重分布")
    ax3.set_xlabel("经度")
    ax3.set_ylabel("纬度")
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 确保文件名与你之前保存的一致
    FILE_PATH = r'D:\see-dn\sse-dn_AGCRN_new\spatial_weights.joblib'
    visualize_spatial_weights(FILE_PATH, station_idx=20) # 你可以换不同的station_idx看看