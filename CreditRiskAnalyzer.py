import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# 1. Load the dataset
df = pd.read_csv('cs-training.csv', index_col=0)

# 2. Rename target column
df.rename(columns={'SeriousDlqin2yrs': 'default'}, inplace=True)

# 3. Handle missing values
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
df['NumberOfDependents'].fillna(0, inplace=True)

# 4. Remove outliers (optional)
df = df[df['RevolvingUtilizationOfUnsecuredLines'] < 1.5]
df = df[df['DebtRatio'] < 10000]

# 5. Split features and target
X = df.drop('default', axis=1)
y = df['default']

# 6. Resample with SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# 8. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Train models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    print(f"\nModel: {name}")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC Score:", auc)
    results[name] = {
        "f1": classification_report(y_test, y_pred, output_dict=True)["1"]["f1-score"],
        "auc": auc
    }
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 10. Display summary
print("\nSummary of Results:")
for model, metrics in results.items():
    print(f"{model} - F1 Score: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
