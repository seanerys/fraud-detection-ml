import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             RocCurveDisplay, PrecisionRecallDisplay)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Scale Amount and Time (V1-V28 already PCA scaled)
scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

# ── 2. TRAIN/TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"Train fraud cases: {y_train.sum()} | Test fraud cases: {y_test.sum()}")

# ── 3. DEFINE MODELS ──────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost':             XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric='logloss', verbosity=0),
    'LightGBM':            LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
}

models_weighted = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42,
                                               class_weight='balanced'),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42,
                                                   class_weight='balanced'),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42,
                                                   class_weight='balanced', n_jobs=-1),
    'XGBoost':             XGBClassifier(n_estimators=100, random_state=42,
                                         scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                         eval_metric='logloss', verbosity=0),
    'LightGBM':            LGBMClassifier(n_estimators=100, random_state=42,
                                          class_weight='balanced', verbose=-1),
}

# ── 4. COST MATRIX ────────────────────────────────────────────────────────────
COST_FP = 10    # cost of flagging a legitimate transaction as fraud
COST_FN = 500   # cost of missing a fraud (much higher)

def expected_cost(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return fp * COST_FP + fn * COST_FN

# ── 5. EVALUATION FUNCTION ────────────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te, threshold=0.5):
    start = time.time()
    model.fit(X_tr, y_tr)
    train_time = round(time.time() - start, 2)

    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        'Model':        name,
        'Accuracy':     round(accuracy_score(y_te, y_pred), 4),
        'Precision':    round(precision_score(y_te, y_pred, zero_division=0), 4),
        'Recall':       round(recall_score(y_te, y_pred, zero_division=0), 4),
        'F1':           round(f1_score(y_te, y_pred, zero_division=0), 4),
        'ROC-AUC':      round(roc_auc_score(y_te, y_prob), 4),
        'PR-AUC':       round(average_precision_score(y_te, y_prob), 4),
        'Exp. Cost':    expected_cost(y_te, y_pred),
        'Train Time(s)':train_time,
    }

# ── 6. RUN ALL BALANCING STRATEGIES ──────────────────────────────────────────
print("\nRunning experiments...")
all_results = []

# Strategy 1: No balancing
print("  [1/4] No balancing...")
for name, model in models.items():
    all_results.append({**evaluate(name, model, X_train, y_train, X_test, y_test),
                        'Balancing': 'None'})

# Strategy 2: Class weighting
print("  [2/4] Class weighting...")
for name, model in models_weighted.items():
    all_results.append({**evaluate(name, model, X_train, y_train, X_test, y_test),
                        'Balancing': 'Class Weight'})

# Strategy 3: SMOTE
print("  [3/4] SMOTE...")
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
for name, model in models.items():
    all_results.append({**evaluate(name, model, X_sm, y_sm, X_test, y_test),
                        'Balancing': 'SMOTE'})

# Strategy 4: ADASYN
print("  [4/4] ADASYN...")
adasyn = ADASYN(random_state=42)
X_ad, y_ad = adasyn.fit_resample(X_train, y_train)
for name, model in models.items():
    all_results.append({**evaluate(name, model, X_ad, y_ad, X_test, y_test),
                        'Balancing': 'ADASYN'})

# ── 7. RESULTS TABLE ──────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
cols = ['Balancing','Model','Accuracy','Precision','Recall','F1',
        'ROC-AUC','PR-AUC','Exp. Cost','Train Time(s)']
results_df = results_df[cols].sort_values(['Balancing','PR-AUC'], ascending=[True, False])
print("\n" + "="*100)
print("CONSOLIDATED RESULTS TABLE")
print("="*100)
print(results_df.to_string(index=False))
results_df.to_csv('results.csv', index=False)
print("\nSaved to results.csv")

# ── 8. CONFUSION MATRICES ─────────────────────────────────────────────────────
print("\nPlotting confusion matrices...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
smote_models = {name: model for name, model in models.items()}
for idx, (name, model) in enumerate(smote_models.items()):
    if idx >= 5: break
    model.fit(X_sm, y_sm)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues',
                xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'])
    axes[idx].set_title(f'{name}\n(SMOTE)')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')
axes[5].axis('off')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()

# ── 9. ROC CURVES ─────────────────────────────────────────────────────────────
print("Plotting ROC curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for name, model in models.items():
    model.fit(X_sm, y_sm)
    y_prob = model.predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=axes[0])
axes[0].set_title('ROC Curves (SMOTE)')
axes[0].plot([0,1],[0,1],'k--')

for name, model in models.items():
    model.fit(X_sm, y_sm)
    y_prob = model.predict_proba(X_test)[:, 1]
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, name=name, ax=axes[1])
axes[1].set_title('Precision-Recall Curves (SMOTE)')

plt.tight_layout()
plt.savefig('roc_pr_curves.png', dpi=150)
plt.show()

# ── 10. THRESHOLD TUNING ──────────────────────────────────────────────────────
print("\nThreshold tuning analysis...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42,
                                   class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.05)
threshold_results = []
for t in thresholds:
    y_pred_t = (y_prob_rf >= t).astype(int)
    threshold_results.append({
        'Threshold': round(t, 2),
        'Precision': round(precision_score(y_test, y_pred_t, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, y_pred_t, zero_division=0), 4),
        'F1':        round(f1_score(y_test, y_pred_t, zero_division=0), 4),
        'Exp. Cost': expected_cost(y_test, y_pred_t),
    })

thresh_df = pd.DataFrame(threshold_results)
print("\nThreshold Tuning – Random Forest (Class Weighted):")
print(thresh_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(thresh_df['Threshold'], thresh_df['Precision'], label='Precision', marker='o')
axes[0].plot(thresh_df['Threshold'], thresh_df['Recall'], label='Recall', marker='o')
axes[0].plot(thresh_df['Threshold'], thresh_df['F1'], label='F1', marker='o')
axes[0].set_xlabel('Threshold')
axes[0].set_title('Precision / Recall / F1 vs Threshold (Random Forest)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(thresh_df['Threshold'], thresh_df['Exp. Cost'], color='crimson', marker='o')
axes[1].set_xlabel('Threshold')
axes[1].set_ylabel('Expected Cost ($)')
axes[1].set_title('Expected Cost vs Threshold (Random Forest)')
axes[1].grid(True)
plt.tight_layout()
plt.savefig('threshold_tuning.png', dpi=150)
plt.show()

# ── 11. K-FOLD CROSS VALIDATION ───────────────────────────────────────────────
print("\nRunning k-fold cross validation (this may take a few minutes)...")
cv_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42,
                                               class_weight='balanced'),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42,
                                                   class_weight='balanced', n_jobs=-1),
    'XGBoost':             XGBClassifier(n_estimators=100, random_state=42,
                                         scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                         eval_metric='logloss', verbosity=0),
    'LightGBM':            LGBMClassifier(n_estimators=100, random_state=42,
                                          class_weight='balanced', verbose=-1),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, model in cv_models.items():
    print(f"  CV: {name}...")
    scores = cross_validate(model, X_train, y_train, cv=skf,
                            scoring=['f1', 'roc_auc', 'average_precision', 'recall'],
                            return_train_score=True, n_jobs=-1)
    cv_results.append({
        'Model':           name,
        'F1 Mean':         round(scores['test_f1'].mean(), 4),
        'F1 Std':          round(scores['test_f1'].std(), 4),
        'ROC-AUC Mean':    round(scores['test_roc_auc'].mean(), 4),
        'ROC-AUC Std':     round(scores['test_roc_auc'].std(), 4),
        'PR-AUC Mean':     round(scores['test_average_precision'].mean(), 4),
        'PR-AUC Std':      round(scores['test_average_precision'].std(), 4),
        'Recall Mean':     round(scores['test_recall'].mean(), 4),
        'Recall Std':      round(scores['test_recall'].std(), 4),
        'Train F1 Mean':   round(scores['train_f1'].mean(), 4),
    })

cv_df = pd.DataFrame(cv_results)
print("\nK-FOLD CROSS VALIDATION RESULTS (5-fold, Class Weighted):")
print(cv_df.to_string(index=False))
cv_df.to_csv('cv_results.csv', index=False)

# ── 12. FEATURE IMPORTANCE ────────────────────────────────────────────────────
print("\nPlotting feature importance...")
rf_final = RandomForestClassifier(n_estimators=100, random_state=42,
                                   class_weight='balanced', n_jobs=-1)
rf_final.fit(X_sm, y_sm)
importances = pd.Series(rf_final.feature_importances_, index=X.columns)
top_features = importances.nlargest(15)

plt.figure(figsize=(10, 6))
top_features.sort_values().plot(kind='barh', color='steelblue')
plt.title('Top 15 Feature Importances – Random Forest (SMOTE)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

print("\n✓ All experiments complete!")
print("Output files: results.csv, cv_results.csv")
print("Plots: confusion_matrices.png, roc_pr_curves.png, threshold_tuning.png, feature_importance.png")

