# -*- coding: utf-8 -*-
"""
特征选择脚本 (2/2): 为“岩石专家”模型筛选最佳特征 
"""
import pandas as pd
import numpy as np
import time
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# ======================= 1. 用户配置区 =======================
MULTISCALE_DATASET_PATH = "path"
OUTPUT_ROCKEXPERT_CSV = "path"
RANDOM_STATE = 42;
VEGETATION_LABEL = 3;
CV_FOLDS = 5;
STEP = 0.1


# ======================= 2. 主执行流程 =======================
def main_rockexpert():
    print("--- “岩石专家”特征选择脚本启动 ) ---")

    print(f"步骤 1: 加载多尺度数据集: {MULTISCALE_DATASET_PATH}")
    df_multi = pd.read_csv(MULTISCALE_DATASET_PATH)
    print(f"多尺度数据集加载成功，形状: {df_multi.shape}")

    print("\n步骤 2: 动态计算差值/比值特征 ")
    X_all = df_multi.drop(columns=['label', 'filename'], errors='ignore')

    new_features_list = []  # 1. 创建一个空列表
    epsilon = 1e-6

    # 2. 循环计算并添加到列表
    for col_10cm in [c for c in X_all.columns if '_10cm' in c]:
        col_30cm = col_10cm.replace('_10cm', '_30cm')
        if col_30cm in X_all.columns:
            diff_col_name = col_10cm.replace('_10cm', '_diff')
            diff_series = X_all[col_30cm] - X_all[col_10cm]
            diff_series.name = diff_col_name
            new_features_list.append(diff_series)

            ratio_col_name = col_10cm.replace('_10cm', '_ratio')
            ratio_series = X_all[col_10cm] / (X_all[col_30cm] + epsilon)
            ratio_series.name = ratio_col_name
            new_features_list.append(ratio_series)

    # 3. 一次性合并
    if new_features_list:
        X_all = pd.concat([X_all] + new_features_list, axis=1)

    print(f"终极特征池构建完毕，总特征数: {X_all.shape[1]}")

    df_full_features = X_all.copy()
    df_full_features['label'] = df_multi['label']

    df_rock = df_full_features[df_full_features['label'] != VEGETATION_LABEL].copy()
    print(f"已筛选出岩石样本，用于特征选择。岩石样本数量: {len(df_rock)}")

    X_rock = df_rock.drop(columns=['label', 'filename'], errors='ignore')
    y_rock = df_rock['label']

    print("\n步骤 3: 配置并启动 RFECV 特征选择...")
    estimator = RandomForestClassifier(n_estimators=150, max_depth=20, class_weight='balanced',
                                       random_state=RANDOM_STATE, n_jobs=-1)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    selector = RFECV(estimator=estimator, step=STEP, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)

    start_time = time.time()
    selector.fit(X_rock, y_rock)
    end_time = time.time()

    print(f"\n特征选择完成！耗时: {end_time - start_time:.2f} 秒")

    print("\n步骤 4: 分析结果并保存新的数据集...")
    print(f"RFECV 已自动找到最佳特征数量为: {selector.n_features_}")

    optimal_features = X_rock.columns[selector.support_].tolist()

    print("\n被选中的“岩石专家黄金特征”列表:")
    for i in range(0, len(optimal_features), 5): print("  ", optimal_features[i:i + 5])

    df_optimal = X_all[optimal_features].copy()
    df_optimal['label'] = df_multi['label']
    if 'filename' in df_multi.columns: df_optimal['filename'] = df_multi['filename']

    df_optimal.to_csv(OUTPUT_ROCKEXPERT_CSV, index=False)

    print(f"\n--- 成功！---")
    print(f"新的“岩石专家”最优特征集已保存至: {OUTPUT_ROCKEXPERT_CSV}")


if __name__ == "__main__":
    main_rockexpert()