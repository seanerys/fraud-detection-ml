import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

cal_models = {
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

plt.figure(figsize=(10, 7))
plt.plot([0,1],[0,1], 'k--', label='Perfectly calibrated')

for name, model in cal_models.items():
    print(f"Fitting {name}...")
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='quantile')
    plt.plot(prob_pred, prob_true, marker='o', label=name)

plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives (Actual Fraud Rate)')
plt.title('Calibration Curves – All Models (Class Weighted)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('calibration_curves.png', dpi=150)
plt.show()
print("Done!")