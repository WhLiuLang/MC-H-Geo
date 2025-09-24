# -*- coding: utf-8 -*-
"""
最终实验脚本: 两步走地质后处理 (Z轴分层 + 局部平滑)

功能:
1.  【第一步 - 宏观修正】执行Z轴地层扫描与置信度加权智能修正，建立地层主体框架。
2.  【第二步 - 介观平滑】在宏观修正结果的基础上，再执行一次基于局部邻域的置信度加权平滑，
    对岩性边界和细节进行最终的“打磨”。
3.  评估每一步处理带来的性能变化。
"""
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial import cKDTree

# ======================= 1. 用户配置区 =======================
# --- 输入路径 ---
# 请确保这个预测文件包含 'filename', 'predicted_label', 'confidence' 三列
PREDICTIONS_CSV = "path"
CENTROIDS_CSV = "path"
GROUND_TRUTH_CSV = "path"

# --- 输出路径 ---
OUTPUT_DIR = "final_postprocessing_files"
OUTPUT_SMOOTHED_CSV = os.path.join(OUTPUT_DIR, "final_two_step_smoothed_predictions.csv")

# --- 算法参数 ---
VEGETATION_LABEL = 3
# --- 阶段一：Z轴分层参数 ---
SWEEP_STEP = 0.1
LAYER_TOLERANCE = 0.05
# --- 阶段二：局部平滑参数 ---
LOCAL_SMOOTHING_RADIUS = 0.5 


# ======================= 2. 主执行流程 =======================
def main():
    print("--- 终极两步走后处理脚本启动 ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 步骤 1: 加载所有必需文件 ---
    print("步骤 1: 加载预测、坐标和真实标签...")
    try:
        df_preds = pd.read_csv(PREDICTIONS_CSV, encoding='utf-8-sig')
        df_centroids = pd.read_csv(CENTROIDS_CSV, encoding='utf-8-sig')
        df_truth = pd.read_csv(GROUND_TRUTH_CSV, usecols=['filename', 'label'], encoding='gbk')
        df_truth['label'] = np.where(df_truth['label'] == 4, 0, df_truth['label'])
    except Exception as e:
        print(f"错误: 读取文件时出错。 {e}"); return

    if 'confidence' not in df_preds.columns:
        print("错误: 输入的预测文件 PREDICTIONS_CSV 中缺少 'confidence' 列！");
        return

    df_full = pd.merge(df_centroids, df_preds, on='filename')
    print(f"数据加载成功，总计 {len(df_full)} 个体素。")

    # ======================= 阶段一：Z轴地层扫描宏观修正 =======================
    print("\n--- 阶段一：执行Z轴地层扫描宏观修正 ---")

    # 初始化一个新列来存储第一步修正的结果
    df_full['strata_smoothed_label'] = df_full['predicted_label']
    min_z, max_z = df_full['z'].min(), df_full['z'].max()

    height_range = np.arange(max_z, min_z, -SWEEP_STEP)
    for z_height in tqdm(height_range, desc="Phase 1: Strata Sweeping"):
        layer_mask = df_full['z'].between(z_height - LAYER_TOLERANCE, z_height + LAYER_TOLERANCE)
        layer_df = df_full[layer_mask]
        if len(layer_df) < 2: continue

        votes = layer_df.groupby('predicted_label')['confidence'].sum()
        rock_votes = votes.drop(VEGETATION_LABEL, errors='ignore')

        if not rock_votes.empty:
            dominant_lithology = rock_votes.idxmax()
            correction_mask = (layer_df['predicted_label'] != dominant_lithology) & (
                        layer_df['predicted_label'] != VEGETATION_LABEL)
            indices_to_correct = layer_df[correction_mask].index
            df_full.loc[indices_to_correct, 'strata_smoothed_label'] = dominant_lithology

    # ======================= 阶段二：局部邻域平滑“打磨” =======================
    print("\n--- 阶段二：执行局部邻域平滑“打磨” ---")

    # 构建空间索引
    coordinates = df_full[['x', 'y', 'z']].values
    kdtree = cKDTree(coordinates)

    # 找到每个点的空间邻居
    neighbor_indices_list = kdtree.query_ball_tree(kdtree, r=LOCAL_SMOOTHING_RADIUS)

    # 初始化最终标签列
    final_labels = []

    # 在第一步修正结果的基础上进行投票
    for i in tqdm(range(len(df_full)), desc="Phase 2: Local Smoothing"):
        neighbor_indices = neighbor_indices_list[i]

        # 投票的“选票”来自于第一阶段修正后的标签
        neighbor_labels = df_full.iloc[neighbor_indices]['strata_smoothed_label'].values

        if len(neighbor_labels) > 0:
            most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
            final_labels.append(most_common_label)
        else:
            # 如果没有邻居，则保留第一阶段的结果
            final_labels.append(df_full.iloc[i]['strata_smoothed_label'])

    df_full['final_smoothed_label'] = final_labels

    # ======================= 评估与保存 =======================
    print("\n--- 最终性能评估 ---")
    df_eval = pd.merge(df_full, df_truth, on='filename', how='inner')
    if not df_eval.empty:
        y_true = df_eval['label'].values
        y_pred_before = df_eval['predicted_label'].values
        y_pred_strata = df_eval['strata_smoothed_label'].values
        y_pred_final = df_eval['final_smoothed_label'].values

        acc_before = accuracy_score(y_true, y_pred_before)
        acc_strata = accuracy_score(y_true, y_pred_strata)
        acc_final = accuracy_score(y_true, y_pred_final)

        print("\n--- 性能对比 ---")
        print(f"初步预测 - 准确率 (Accuracy): {acc_before:.4f}")
        print(f"阶段一后 - 准确率 (Accuracy): {acc_strata:.4f}")
        print(f"阶段二后 - 准确率 (Accuracy): {acc_final:.4f}")

        gain_1 = acc_strata - acc_before
        gain_2 = acc_final - acc_strata
        total_gain = acc_final - acc_before
        print(f"\n>>> 阶段一提升: {gain_1:+.4f}")
        print(f">>> 阶段二提升: {gain_2:+.4f}")
        if acc_before > 0: print(f">>> 总提升: {total_gain:+.4f} ({(total_gain / acc_before):+.2%})")

    df_full.to_csv(OUTPUT_SMOOTHED_CSV, index=False, encoding='utf-8-sig')
    print(f"\n🎉🎉🎉 最终两步走平滑结果已保存至: {OUTPUT_SMOOTHED_CSV} 🎉🎉🎉")


if __name__ == "__main__":
    main()