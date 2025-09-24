# -*- coding: utf-8 -*-
"""
最终的层次化模型训练脚本 (使用最优特征集)

功能:
1.  分别加载为“守门员”和“岩石专家”筛选出的最优特征集。
2.  不再进行任何动态特征工程，直接使用筛选好的特征。
3.  训练最终的、高度优化的层次化模型，并进行评估。
"""
import os
import time
import csv
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings("ignore")

# ======================= 1. 用户配置区 =======================
# --- 输入文件路径 (关键修改) ---
# 指向您筛选出的两个最优特征集
GOALKEEPER_DATA_CSV = "D:/研究生/毕业论文数据/前王家河3号露头/岩性标准样/标准体素样本/goalkeeper_optimal_features_1.csv"
ROCKEXPERT_DATA_CSV = "D:/研究生/毕业论文数据/前王家河3号露头/岩性标准样/标准体素样本/rockexpert_optimal_features_1.csv"

# --- 输出与参数配置 ---
OUTPUT_DIR = "final_model_results"
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "final_model_summary_modified.pdf")
RANDOM_STATE = 42
LABELS = ["mudstone", "sandstone", "siltstone", "vegetation"]
N_CLASSES = len(LABELS)
VEGETATION_LABEL = 3
BASE_LIST = ["RF", "XGB", "MLP"]
STACKING_FOLDS = 5


# ======================= 2. 模型与辅助函数 (与之前版本一致) =======================
def ensure_1d(y): return np.asarray(y).ravel()


def build_branches(df):
    feat_cols = [c for c in df.columns if c not in ['label', 'filename']]
    numeric_cols = [c for c in feat_cols if np.issubdtype(df[c].dtype, np.number)]
    patterns = {'geom': [r'roughness', r'linearity', r'planarity', r'sphericity', r'point_den', r'geom_pca'],
                'spectral': [r'ref_', r'refnorm_', r'amp_'], 'texture': [r'glcm_', r'fft_', r'tex_feat']}
    branches = {}
    for bname, pat_list in patterns.items():
        branch_cols = [c for c in numeric_cols if any(re.search(p, c) for p in pat_list)]
        if branch_cols: branches[bname] = branch_cols
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


def plot_cm(cm, title, out_png, labels_list):
    plt.figure(figsize=(6, 5));
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_list, yticklabels=labels_list)
    plt.xlabel("Predicted");
    plt.ylabel("Actual");
    plt.title(title);
    plt.tight_layout();
    plt.savefig(out_png, dpi=150);
    plt.close()


# ======================= 3. 核心训练与预测函数 (最终版) =======================
def train_goalkeeper(X_df, y):
    print("\n===== STAGE 1: Training Goalkeeper (on Optimal Features) =====")
    y_binary = np.where(y == VEGETATION_LABEL, 1, 0)
    print(f"Goalkeeper training data: X shape={X_df.shape}, y_binary distribution={Counter(y_binary)}")
    model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=3, class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_df.values, y_binary)
    print("✅ Goalkeeper (Random Forest) trained.")
    return model


def train_rock_expert(X_expert_df, y):
    print("\n===== STAGE 2: Training Rock Expert (on Optimal Features) =====")
    rock_indices = np.where(y != VEGETATION_LABEL)[0]
    X_rock = X_expert_df.iloc[rock_indices].reset_index(drop=True)
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


def predict_hierarchical(X_goalkeeper_df, X_expert_df, goalkeeper, rock_expert_bundle):
    goalkeeper_preds = goalkeeper.predict(X_goalkeeper_df.values)
    final_preds = np.full(len(X_goalkeeper_df), -1, dtype=int)
    veg_indices = np.where(goalkeeper_preds == 1)[0];
    final_preds[veg_indices] = VEGETATION_LABEL
    rock_indices = np.where(goalkeeper_preds == 0)[0]
    if len(rock_indices) > 0 and rock_expert_bundle is not None:
        X_rock_to_predict = X_expert_df.iloc[rock_indices]
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
        expert_preds_enc = meta_learner.predict(meta_features_test)
        final_preds[rock_indices] = le.inverse_transform(expert_preds_enc)
    return final_preds


# ======================= 4. 主执行流程 (最终版) =======================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📥 Loading Final Optimal Datasets...")
    # --- 加载两个独立的、最优化的数据集 ---
    try:
        # <<< --- 核心修改：为 read_csv 添加 encoding='gbk' 参数 --- >>>
        df_goalkeeper = pd.read_csv(GOALKEEPER_DATA_CSV, encoding='gbk')
        df_rockexpert = pd.read_csv(ROCKEXPERT_DATA_CSV, encoding='gbk')
        # <<< --- 修改结束 --- >>>

        print("训练数据加载成功。")
    except FileNotFoundError as e:
        print(f"加载训练集时出错: {e}");
        return
    except UnicodeDecodeError:
        print(f"使用'gbk'编码读取失败，尝试使用'utf-8'...")
        try:
            # 如果gbk失败，也尝试一下utf-8，增加稳健性
            df_goalkeeper = pd.read_csv(GOALKEEPER_DATA_CSV, encoding='utf-8')
            df_rockexpert = pd.read_csv(ROCKEXPERT_DATA_CSV, encoding='utf-8')
        except Exception as e:
            print(f"尝试多种编码后，读取训练集仍然失败: {e}");
            return

    # 从任一数据集中提取标签和索引（它们应该是一致的）
    y_all = df_goalkeeper['label'].values
    y_all = np.where(y_all == 4, 0, y_all)  # 确保标签已合并

    X_goalkeeper_df = df_goalkeeper.drop(columns=['label', 'filename'], errors='ignore')
    X_rockexpert_df = df_rockexpert.drop(columns=['label', 'filename'], errors='ignore')

    # 对齐检查
    if len(X_goalkeeper_df) != len(X_rockexpert_df):
        print("Error: The two optimal datasets have different numbers of rows!");
        return

    print(f"Goalkeeper feature set shape: {X_goalkeeper_df.shape}")
    print(f"Rock Expert feature set shape: {X_rockexpert_df.shape}")
    print(f"Label distribution: {Counter(y_all)}")

    # 创建一致的训练/验证集划分索引
    indices = np.arange(len(y_all))
    tr_idx, val_idx, y_tr, y_val = train_test_split(indices, y_all, test_size=0.2, random_state=RANDOM_STATE,
                                                    stratify=y_all)

    # 为每个模型划分对应的数据
    X_tr_goalkeeper, X_val_goalkeeper = X_goalkeeper_df.iloc[tr_idx], X_goalkeeper_df.iloc[val_idx]
    X_tr_rockexpert, X_val_rockexpert = X_rockexpert_df.iloc[tr_idx], X_rockexpert_df.iloc[val_idx]

    print(f"\nTraining set size: {len(y_tr)}, Validation set size: {len(y_val)}")

    # ---------- Train Hierarchical Model ----------
    t_train_start = time.time()

    goalkeeper_model = train_goalkeeper(X_tr_goalkeeper, y_tr)
    rock_expert_bundle = train_rock_expert(X_tr_rockexpert, y_tr)

    train_dt = time.time() - t_train_start
    print(f"\nTotal training time: {train_dt:.2f} seconds")

    # ---------- Evaluate on Validation Set ----------
    print("\n🔬 Evaluating on validation set...")
    y_pred_val = predict_hierarchical(X_val_goalkeeper, X_val_rockexpert, goalkeeper_model, rock_expert_bundle)

    val_acc = accuracy_score(y_val, y_pred_val)
    val_f1m = f1_score(y_val, y_pred_val, average='macro')

    print(f"\n>>> FINAL Validation Accuracy: {val_acc:.4f}")
    print(f">>> FINAL Validation Macro-F1: {val_f1m:.4f}")

    # ---------- Reporting ----------
    cm_val = confusion_matrix(y_val, y_pred_val, labels=np.arange(N_CLASSES))
    cm_png = os.path.join(OUTPUT_DIR, "Final_Model_Validation_CM.png")
    plot_cm(cm_val, f"Final Hierarchical Model (Validation Acc: {val_acc:.3f})", cm_png, LABELS)

    # ... CSV and PDF reporting code (与之前版本一致) ...
    csv_file = os.path.join(OUTPUT_DIR, "final_model_results.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f);
        writer.writerow(["Model", "Validation-Accuracy", "Validation-Macro-F1", "Training-Time(s)"])
        writer.writerow(["Hierarchical_Optimized", f"{val_acc:.4f}", f"{val_f1m:.4f}", f"{train_dt:.2f}"])
    print(f"\nCSV report saved to {csv_file}")

    styles = getSampleStyleSheet();
    doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4)
    story = [Paragraph("Final Optimized Model Report", styles["Title"]), Spacer(1, 12),
             Paragraph(f"Validation Accuracy: {val_acc:.4f}", styles["h2"]),
             Paragraph(f"Validation Macro-F1 Score: {val_f1m:.4f}", styles["h2"]), Spacer(1, 12),
             Paragraph("Validation Confusion Matrix:", styles["h3"]), Image(cm_png, width=400, height=320)]
    doc.build(story);
    print(f"PDF report saved to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()