# -*- coding: utf-8 -*-
"""
最终实验脚本 : 三层上下文感知智能修正后处理器

"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from collections import Counter
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA

# ======================= 1. 用户配置区 =======================
# --- 输入路径 ---
PREDICTIONS_CSV = "path"
CENTROIDS_CSV = "path"
GROUND_TRUTH_CSV = "path"

# 手动选取的参考平面点文件 (逗号分隔的txt) ---
STRATIGRAPHIC_PLANE_POINTS_FILE = "path"

# --- 输出路径 ---
OUTPUT_DIR = "final_postprocessing_files"
OUTPUT_SMOOTHED_CSV = os.path.join(OUTPUT_DIR, "scene_predictions_smoothed.csv")

# --- 算法参数 ---
SWEEP_STEP = 0.10;
LAYER_TOLERANCE = 0.05;
VEGETATION_LABEL = 3
LABELS_MAP = {0: "mudstone", 1: "sandstone", 2: "siltstone", 3: "vegetation"}


# ======================= 2. 核心函数  =======================
def fit_plane_to_points(points):
    pca = PCA(n_components=3);
    pca.fit(points)
    normal_vector = pca.components_[2]
    point_on_plane = np.mean(points, axis=0)
    return normal_vector, point_on_plane


def project_points_to_normal(points, normal, reference_point):
    return np.dot(points - reference_point, normal)


# ======================= 3. 主执行流程 =======================
def main():
    print("--- 三层上下文感知智能修正脚本启动 ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("步骤 1: 加载预测、坐标、真实标签和参考平面点...")
    try:
        df_preds = pd.read_csv(PREDICTIONS_CSV, encoding='utf-8-sig')
        df_centroids = pd.read_csv(CENTROIDS_CSV, encoding='utf-8-sig')
        df_truth = pd.read_csv(GROUND_TRUTH_CSV, usecols=['filename', 'label'], encoding='gbk')
        df_truth['label'] = np.where(df_truth['label'] == 4, 0, df_truth['label'])

        df_plane_points = pd.read_csv(
            STRATIGRAPHIC_PLANE_POINTS_FILE,
            header=None,
            names=['x', 'y', 'z'],
            sep=',', 
            usecols=[0, 1, 2]  # 确保只读取前三列
        )

    except Exception as e:
        print(f"错误: 读取文件时出错。 {e}");
        return

    df_full = pd.merge(df_centroids, df_preds, on='filename')
    print(f"数据加载成功，总计 {len(df_full)} 个体素。")

    print("\n步骤 2: 拟合主地质层面...")
    plane_points = df_plane_points[['x', 'y', 'z']].values
    if len(plane_points) < 3:
        print("错误: 参考平面点必须至少有3个！");
        return
    plane_normal, point_on_plane = fit_plane_to_points(plane_points)
    print(f"主地层法向量计算完毕: {plane_normal}")

    print("步骤 3: 计算所有体素的“地质高程”...")
    all_coordinates = df_full[['x', 'y', 'z']].values
    df_full['geological_height'] = project_points_to_normal(all_coordinates, plane_normal, point_on_plane)

    print("步骤 4: 开始执行地层扫描与置信度加权智能修正...")
    df_full['final_smoothed_label'] = df_full['predicted_label']
    min_height = df_full['geological_height'].min()
    max_height = df_full['geological_height'].max()

    # 预计算每一层的主旋律岩性
    strata_info = {}
    height_range = np.arange(max_height, min_height, -SWEEP_STEP)
    for i, height in enumerate(tqdm(height_range, desc="Pre-calculating strata dominants")):
        layer_mask = df_full['geological_height'].between(height - LAYER_TOLERANCE, height + LAYER_TOLERANCE)
        layer_df = df_full[layer_mask]
        dominant_lithology = -1  # 使用 -1 代表“未确定”或“纯植被层”
        if len(layer_df) < 2:
            strata_info[i] = {'mask': layer_mask, 'dominant': dominant_lithology}
            continue

        # 按预测标签分组，并对每个组的置信度求和
        confidence_votes = layer_df.groupby('predicted_label')['confidence'].sum()

        # 移除植被的投票
        rock_votes = confidence_votes.drop(VEGETATION_LABEL, errors='ignore')

        if not rock_votes.empty:
            # 找到岩石中置信度总和最高的那个作为主旋律
            dominant_lithology = rock_votes.idxmax()
        strata_info[i] = {'mask': layer_mask, 'dominant': dominant_lithology}

    # 执行三层上下文修正
    for i in tqdm(range(len(height_range)), desc="Applying 3-strata correction"):
        # 这个 if 判断现在可以安全地处理 dominant_lithology == -1 的情况
        if i not in strata_info or strata_info[i]['dominant'] == -1:
            continue

        L_current = strata_info[i]['dominant']
        current_layer_df = df_full[strata_info[i]['mask']]

        correction_mask = (current_layer_df['predicted_label'] != L_current) & \
                          (current_layer_df['predicted_label'] != VEGETATION_LABEL)

        indices_to_correct = current_layer_df[correction_mask].index
        df_full.loc[indices_to_correct, 'final_smoothed_label'] = L_current

    print("\n步骤 5: 评估平滑前后的精度...")
    df_eval = pd.merge(df_full, df_truth, on='filename', how='inner')
    if df_eval.empty:
        print("警告：没有任何一个场景体素能在真实标签文件中找到对应，无法进行精度评估。")
    else:
        y_true, y_pred_before, y_pred_after = df_eval['label'].values, df_eval['predicted_label'].values, df_eval[
            'final_smoothed_label'].values
        acc_before, f1_before = accuracy_score(y_true, y_pred_before), f1_score(y_true, y_pred_before, average='macro')
        acc_after, f1_after = accuracy_score(y_true, y_pred_after), f1_score(y_true, y_pred_after, average='macro')
        print("\n--- 性能对比 ---")
        print(f"修正前 - 准确率: {acc_before:.4f}, 宏平均F1: {f1_before:.4f}")
        print(f"修正后 - 准确率: {acc_after:.4f}, 宏平均F1: {f1_after:.4f}")
        acc_gain = acc_after - acc_before
        print(f"\n>>> 准确率提升: {acc_gain:+.4f} ({(acc_gain / acc_before):+.2%})")

    df_full.to_csv(OUTPUT_SMOOTHED_CSV, index=False, encoding='utf-8-sig')
    print(f"\n🎉🎉🎉 最终地质智能修正结果已保存至: {OUTPUT_SMOOTHED_CSV} 🎉🎉🎉")


if __name__ == "__main__":
    main()