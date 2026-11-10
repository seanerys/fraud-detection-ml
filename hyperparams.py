import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, make_scorer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
import pandas as pd
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)

pr_auc = make_scorer(average_precision_score, needs_proba=True)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# XGBoost grid search
print("\nXGBoost grid search (this may take a few minutes)...")
xgb_params = {
    'max_depth':        [3, 5, 7],
    'min_child_weight': [1, 5, 10],
    'reg_lambda':       [1, 5, 10],
}
xgb = XGBClassifier(n_estimators=100, random_state=42,
                    scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                    eval_metric='logloss', verbosity=0)
xgb_gs = GridSearchCV(xgb, xgb_params, scoring='average_precision',
                      cv=skf, n_jobs=-1, verbose=1)
xgb_gs.fit(X_sm, y_sm)
print(f"Best XGBoost params: {xgb_gs.best_params_}")
print(f"Best XGBoost PR-AUC (CV): {xgb_gs.best_score_:.4f}")

# LightGBM grid search
print("\nLightGBM grid search...")
lgb_params = {
    'max_depth':    [3, 5, 7],
    'min_child_samples': [10, 30, 50],
    'reg_lambda':   [1, 5, 10],
}
lgb = LGBMClassifier(n_estimators=100, random_state=42,
                     class_weight='balanced', verbose=-1)
lgb_gs = GridSearchCV(lgb, lgb_params, scoring='average_precision',
                      cv=skf, n_jobs=-1, verbose=1)
lgb_gs.fit(X_sm, y_sm)
print(f"Best LightGBM params: {lgb_gs.best_params_}")
print(f"Best LightGBM PR-AUC (CV): {lgb_gs.best_score_:.4f}")

# Now compare default vs tuned
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
print("\nComparing default vs tuned models...")

results = []
for name, default, tuned in [
    ("XGBoost",
     XGBClassifier(n_estimators=100, random_state=42,
                   scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                   eval_metric='logloss', verbosity=0),
     xgb_gs.best_estimator_),
    ("LightGBM",
     LGBMClassifier(n_estimators=100, random_state=42,
                    class_weight='balanced', verbose=-1),
     lgb_gs.best_estimator_),
]:
    for label, model in [("Default", default), ("Tuned", tuned)]:
        model.fit(X_sm, y_sm)
        y_prob = model.predict_proba(X_test)[:,1]
        y_pred = (y_prob >= 0.5).astype(int)
        # train score for overfitting check
        y_train_pred = model.predict(X_sm)
        results.append({
            'Model': f"{name} ({label})",
            'PR-AUC': round(average_precision_score(y_test, y_prob), 4),
            'F1':     round(f1_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'Train F1': round(f1_score(y_sm, y_train_pred), 4),
        })

df_res = pd.DataFrame(results)
print("\nDefault vs Tuned Comparison:")
print(df_res.to_string(index=False))
df_res.to_csv('hyperparam_results.csv', index=False)
print("\nDone! Saved to hyperparam_results.csv")