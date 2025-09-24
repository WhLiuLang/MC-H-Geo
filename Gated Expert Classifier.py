# -*- coding: utf-8 -*-
"""
最终实验脚本 : 最终模型训练与全场景初步预测 
"""
import os
import time
import warnings
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ======================= 1. 用户配置区 =======================
TRAIN_GOALKEEPER_CSV = "path"
TRAIN_ROCKEXPERT_CSV = "path"
SCENE_GOALKEEPER_CSV = "path"
SCENE_ROCKEXPERT_CSV = "path"
OUTPUT_DIR = "final_postprocessing_files"
RANDOM_STATE = 42;
LABELS = ["mudstone", "sandstone", "siltstone", "vegetation"];
N_CLASSES = len(LABELS)
VEGETATION_LABEL = 3;
BASE_LIST = ["RF", "XGB", "MLP"];
STACKING_FOLDS = 5


# ======================= 2. 模型与辅助函数  =======================
def build_branches(df):
    feat_cols = [c for c in df.columns if c not in ['label', 'filename']];
    numeric_cols = [c for c in feat_cols if np.issubdtype(df[c].dtype, np.number)]
    patterns = {'geom': [r'roughness', r'linearity', r'planarity', r'sphericity', r'point_den', r'geom_pca'],
                'spectral': [r'ref_', r'refnorm_', r'amp_'], 'texture': [r'glcm_', r'fft_', r'tex_feat']}
    branches = {};
    [branches.update({bname: [c for c in numeric_cols if any(re.search(p, c) for p in pat_list)]}) for bname, pat_list
     in patterns.items() if [c for c in numeric_cols if any(re.search(p, c) for p in pat_list)]]
    if not branches: branches['full_features'] = numeric_cols
    return branches


def get_base_model(name):
    if name == "RF": return RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=1,
                                                   max_features="sqrt", class_weight="balanced_subsample",
                                                   random_state=RANDOM_STATE, n_jobs=-1)
    if name == "XGB": return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9,
                                           colsample_bytree=0.9, reg_lambda=6.0, reg_alpha=0.5,
                                           random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="mlogloss",
                                           n_jobs=-1, tree_method="hist")
    if name == "MLP": return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=3e-4, max_iter=600,
                                           random_state=RANDOM_STATE)
    raise ValueError(name)


def train_goalkeeper(X_df, y):
    print("\n===== STAGE 1: Training Goalkeeper (on Corrected Labeled Data) =====")
    y_binary = np.where(y == VEGETATION_LABEL, 1, 0)
    print(f"Goalkeeper training data: X shape={X_df.shape}, y_binary distribution={Counter(y_binary)}")
    model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=3, class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_df.values, y_binary)
    print("✅ Goalkeeper (Random Forest) trained.")
    return model


def train_rock_expert(X_expert_df, y):
    print("\n===== STAGE 2: Training Rock Expert (on Corrected Labeled Data) =====")
    rock_indices = np.where(y != VEGETATION_LABEL)[0]
    X_rock = X_expert_df.iloc[rock_indices].reset_index(drop=True);
    y_rock = y[rock_indices]
    le = LabelEncoder();
    y_rock_enc = le.fit_transform(y_rock)
    print(f"Rock Expert training data: X shape={X_rock.shape}, y_rock distribution={Counter(y_rock)}")
    branches = build_branches(X_rock)
    for bname, cols in branches.items(): print(f"  - Branch '{bname}': {len(cols)} features")
    min_class_count = min(np.bincount(y_rock_enc));
    n_splits = min(STACKING_FOLDS, min_class_count)
    if n_splits < 2: print("⚠️ Not enough samples for stacking."); return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_meta_features = []
    for bname, b_cols in branches.items():
        X_branch = X_rock[b_cols];
        scaler = StandardScaler().fit(X_branch);
        X_branch_scaled = scaler.transform(X_branch)
        for mname in BASE_LIST:
            oof_preds_model = np.zeros((len(X_rock), len(le.classes_)))
            for fold, (tr_idx, te_idx) in enumerate(skf.split(X_branch_scaled, y_rock_enc)):
                X_tr, X_te = X_branch_scaled[tr_idx], X_branch_scaled[te_idx];
                y_tr = y_rock_enc[tr_idx]
                clf = get_base_model(mname);
                sw = compute_sample_weight(class_weight='balanced', y=y_tr)
                try:
                    clf.fit(X_tr, y_tr, sample_weight=sw)
                except TypeError:
                    clf.fit(X_tr, y_tr)
                oof_preds_model[te_idx] = clf.predict_proba(X_te)
            oof_meta_features.append(oof_preds_model)
    meta_features = np.hstack(oof_meta_features)
    meta_learner = LogisticRegression(max_iter=1000, multi_class="multinomial", class_weight="balanced",
                                      random_state=RANDOM_STATE)
    meta_learner.fit(meta_features, y_rock_enc)
    final_base_models = {}
    for bname, b_cols in branches.items():
        X_branch = X_rock[b_cols];
        scaler = StandardScaler().fit(X_branch);
        X_branch_scaled = scaler.transform(X_branch)
        final_base_models[bname] = {'scaler': scaler, 'models': {}}
        for mname in BASE_LIST:
            clf = get_base_model(mname);
            sw = compute_sample_weight(class_weight='balanced', y=y_rock_enc)
            try:
                clf.fit(X_branch_scaled, y_rock_enc, sample_weight=sw)
            except TypeError:
                clf.fit(X_branch_scaled, y_rock_enc)
            final_base_models[bname]['models'][mname] = clf
    print("✅ Rock Expert trained.")
    return {"meta_learner": meta_learner, "base_models": final_base_models, "branches": branches, "label_encoder": le}


# ======================= 3. 核心预测函数  =======================
def predict_hierarchical(X_goalkeeper_df, X_expert_df, goalkeeper, rock_expert_bundle):
    # 初始化最终的预测标签和概率矩阵
    final_preds = np.full(len(X_goalkeeper_df), -1, dtype=int)
    final_probs = np.full((len(X_goalkeeper_df), N_CLASSES), 0.0, dtype=float)

    # --- 守门员阶段 ---
    # 获取守门员的原始概率 [P(Rock), P(Vegetation)]
    goalkeeper_probs = goalkeeper.predict_proba(X_goalkeeper_df.values)
    # 守门员的最终决策 (0=Rock, 1=Vegetation)
    goalkeeper_preds = np.argmax(goalkeeper_probs, axis=1)

    # --- 概率和标签的初步分配 ---
    # 所有样本的植被概率，直接来自于守门员
    final_probs[:, VEGETATION_LABEL] = goalkeeper_probs[:, 1]

    # 将被守门员判定为植被的样本，其最终标签直接设为植被
    veg_indices = np.where(goalkeeper_preds == 1)[0]
    final_preds[veg_indices] = VEGETATION_LABEL

    # --- 岩石专家阶段 ---
    rock_indices = np.where(goalkeeper_preds == 0)[0]
    if len(rock_indices) > 0 and rock_expert_bundle is not None:
        X_rock_to_predict = X_expert_df.iloc[rock_indices]

        # (Stacking预测逻辑保持不变)
        meta_features_to_predict = []
        base_models_bundle, branches = rock_expert_bundle['base_models'], rock_expert_bundle['branches']
        for bname, b_cols in branches.items():
            scaler = base_models_bundle[bname]['scaler']
            X_branch_scaled = scaler.transform(X_rock_to_predict[b_cols])
            for mname in BASE_LIST:
                clf = base_models_bundle[bname]['models'][mname]
                meta_features_to_predict.append(clf.predict_proba(X_branch_scaled))
        meta_features_test = np.hstack(meta_features_to_predict)

        meta_learner, le = rock_expert_bundle['meta_learner'], rock_expert_bundle['label_encoder']

        expert_probs_local = meta_learner.predict_proba(meta_features_test)
        expert_preds_local_enc = np.argmax(expert_probs_local, axis=1)
        final_preds[rock_indices] = le.inverse_transform(expert_preds_local_enc)

        for i, idx in enumerate(rock_indices):
            prob_global = np.zeros(N_CLASSES)
            for j, lab in enumerate(le.classes_):
                prob_global[int(lab)] = expert_probs_local[i, j]

            # 最终的岩石概率 = P(是岩石|守门员) * P(具体是哪种岩石|岩石专家)
            final_probs[idx, :VEGETATION_LABEL] = prob_global[:VEGETATION_LABEL] * goalkeeper_probs[idx, 0]

    return final_preds, final_probs


# ======================= 4. 主执行流程 =======================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 步骤 1: 加载所有数据集 ---
    print("--- 步骤 1: 加载所有必需的数据集 ---")
    try:
        df_tr_gk = pd.read_csv(TRAIN_GOALKEEPER_CSV, encoding='gbk')
        df_tr_re = pd.read_csv(TRAIN_ROCKEXPERT_CSV, encoding='gbk')
        df_scene_gk = pd.read_csv(SCENE_GOALKEEPER_CSV, encoding='utf-8')
        df_scene_re = pd.read_csv(SCENE_ROCKEXPERT_CSV, encoding='utf-8')
    except UnicodeDecodeError:
        print("使用 'gbk' 编码读取失败，将尝试 'utf-8'...")
        try:
            df_tr_gk = pd.read_csv(TRAIN_GOALKEEPER_CSV, encoding='gbk')
            df_tr_re = pd.read_csv(TRAIN_ROCKEXPERT_CSV, encoding='gbk')
            df_scene_gk = pd.read_csv(SCENE_GOALKEEPER_CSV, encoding='utf-8')
            df_scene_re = pd.read_csv(SCENE_ROCKEXPERT_CSV, encoding='utf-8')
        except Exception as e:
            print(f"尝试多种编码后，读取文件仍然失败: {e}"); return
    except FileNotFoundError as e:
        print(f"加载文件时出错: {e}"); return
    print("所有数据集加载成功。")

    # --- 步骤 2: 准备和清洗数据 ---
    print("\n--- 步骤 2: 准备并清洗所有数据 (填充NaN) ---")
    y_tr = df_tr_gk['label'].values;
    y_tr = np.where(y_tr == 4, 0, y_tr)
    X_tr_goalkeeper = df_tr_gk.drop(columns=['label', 'filename'], errors='ignore').fillna(0)
    X_tr_rockexpert = df_tr_re.drop(columns=['label', 'filename'], errors='ignore').fillna(0)
    print("训练数据已准备并清洗。")

    if not df_scene_gk['filename'].equals(df_scene_re['filename']):
        print("错误: 全场景数据集文件名顺序不一致！");
        return
    X_scene_goalkeeper = df_scene_gk.drop(columns=['label', 'filename'], errors='ignore').fillna(0)
    X_scene_rockexpert = df_scene_re.drop(columns=['label', 'filename'], errors='ignore').fillna(0)
    print("全场景数据已准备并清洗。")

    # --- 步骤 3: 训练最终模型 ---
    print("\n--- 步骤 3: 训练最终模型 ---")
    t_train_start = time.time()
    goalkeeper_model = train_goalkeeper(X_tr_goalkeeper, y_tr)
    rock_expert_bundle = train_rock_expert(X_tr_rockexpert, y_tr)
    train_dt = time.time() - t_train_start
    print(f"\nTotal training time: {train_dt:.2f} seconds")

    # --- 步骤 4: 对全场景数据进行预测 ---
    print("\n--- 步骤 4: 对全场景进行预测 ---")
    predictions, probabilities = predict_hierarchical(X_scene_goalkeeper, X_scene_rockexpert, goalkeeper_model,
                                                      rock_expert_bundle)
    confidences = np.max(probabilities, axis=1)

    # --- 步骤 5: 保存初步预测结果 ---
    df_predictions = pd.DataFrame({
        'filename': df_scene_gk['filename'],
        'predicted_label': predictions,
        'confidence': confidences
    })

    OUTPUT_PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "preliminary_predictions_with_confidence.csv")
    df_predictions.to_csv(OUTPUT_PREDICTIONS_CSV, index=False, encoding='utf-8-sig')

    print(f"\n初步预测结果 (含置信度) 已保存至: {OUTPUT_PREDICTIONS_CSV}")
    print("\n🎉🎉🎉 脚本一执行完毕！现在您可以运行最终版的后处理脚本了。🎉🎉🎉")


if __name__ == "__main__":
    main()