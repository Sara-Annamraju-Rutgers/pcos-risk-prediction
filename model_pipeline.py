import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap, joblib, matplotlib.pyplot as plt, os

# ---------- LOAD & CLEAN ----------
df = pd.read_csv("data/lifestyle-pcos.csv")
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("?", "", regex=False)
              .str.replace("/", "_")
               .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)  
)
df = df.dropna().drop_duplicates()

# ---------- TARGET ----------
target = "have_you_been_diagnosed_with_pcos_pcod"   
X = df.drop(columns=[target])
y = df[target]

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- PIPELINE ----------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"        
    ))
])

pipeline.fit(X_train, y_train)

# ---------- EVALUATE ----------
report = classification_report(y_test, pipeline.predict(X_test))
print(report)
os.makedirs("outputs", exist_ok=True)
with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

# ---------- SHAP ----------
explainer = shap.Explainer(pipeline.named_steps["model"], X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_plot.png")

# ---------- SAVE MODEL ----------
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/pcos_model.pkl")
print("✅  Saved model to models/pcos_model.pkl")
