# Re-running code after kernel reset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import shap
import joblib
import os

# Load and clean the dataset
df = pd.read_csv("/mnt/data/lifestyle-pcos.csv")
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("?", "", regex=False)
              .str.replace("/", "_")
              .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
)
df = df.dropna().drop_duplicates()

# Define target and features
target = "have_you_been_diagnosed_with_pcos_pcod"
X = df.drop(columns=[target])
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model with pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)

# Create plots directory
os.makedirs("outputs/plots", exist_ok=True)

# 1. Class Distribution Plot
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Class Distribution")
plt.xlabel("PCOS Diagnosis (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/plots/class_distribution.png")

# 2. Confusion Matrix
y_pred = pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/plots/confusion_matrix.png")

# 3. ROC Curve
y_proba = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/roc_curve.png")

# 4. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/precision_recall_curve.png")

# 5. Feature Importance Plot
model = pipeline.named_steps["model"]
importances = model.feature_importances_
feature_names = X.columns
indices = importances.argsort()[::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices][:10], y=feature_names[indices][:10])
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("outputs/plots/feature_importance.png")

# 6. SHAP Summary Plot
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("outputs/plots/shap_summary.png")

import ace_tools as tools; tools.display_dataframe_to_user(name="Sample Dataset", dataframe=df.head())
