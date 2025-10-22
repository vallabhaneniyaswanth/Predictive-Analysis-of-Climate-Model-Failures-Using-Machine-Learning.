# -----------------------------------------------------
# CLIMATE MODEL FAILURE PREDICTION - COMPLETE PROJECT
# -----------------------------------------------------

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Step 2: Generate Realistic Dataset (800 rows)
np.random.seed(42)
n_samples = 800

temperature = np.random.normal(loc=28, scale=5, size=n_samples)
temperature = np.clip(temperature, 15, 40)

mixing = np.random.normal(loc=0.9, scale=0.2, size=n_samples)
mixing = np.clip(mixing, 0.5, 1.5)

viscosity = np.random.normal(loc=1.4, scale=0.3, size=n_samples)
viscosity = np.clip(viscosity, 0.8, 2.0)

# Failure based on probabilistic rules
failure = []
for t, m, v in zip(temperature, mixing, viscosity):
    prob_fail = 0.2
    if t > 32: prob_fail += 0.25
    if v > 1.6: prob_fail += 0.25
    if m > 1.1: prob_fail += 0.15
    prob_fail = min(prob_fail, 0.95)
    failure.append(np.random.choice([0,1], p=[1-prob_fail, prob_fail]))

df = pd.DataFrame({
    'Temperature': np.round(temperature,2),
    'Mixing': np.round(mixing,2),
    'Viscosity': np.round(viscosity,2),
    'Failure': failure
})

# Save dataset (optional)
df.to_csv("ClimateDataset.csv", index=False)

# Step 3: Feature and Label
X = df[['Temperature','Mixing','Viscosity']]
y = df['Failure']

# Step 4: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Handle Imbalanced Classes using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Step 6: Split Data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Step 7: Define Multiple Models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}
probs = {}  # probabilities for ROC

# Step 8: Train and Evaluate All Models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    if hasattr(model, "predict_proba"):
        probs[name] = model.predict_proba(X_test)[:,1]
    print(f"\n--- {name} ---")
    print("Accuracy:", round(acc,3))
    print(classification_report(y_test, y_pred))

# Step 9: Pick Best Model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n✅ Best Model: {best_model_name} with Accuracy {round(results[best_model_name],3)}")

# -----------------------------------------------------
# 🔹 VISUALIZATIONS (6 in total)
# -----------------------------------------------------

# 1️⃣ Class Distribution
plt.figure(figsize=(5,4))
sns.countplot(x='Failure', data=df)
plt.title("Class Distribution in Original Dataset")
plt.show()

# 2️⃣ Accuracy Comparison
plt.figure(figsize=(8,5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.show()

# 3️⃣ Confusion Matrix for Best Model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 4️⃣ ROC Curves for All Models
plt.figure(figsize=(7,5))
for name, model in models.items():
    if name in probs:
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        auc = roc_auc_score(y_test, probs[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 5️⃣ Feature Importance (Best Model if Tree-Based)
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    features = X.columns
    plt.figure(figsize=(6,4))
    sns.barplot(x=importances, y=features, palette="crest")
    plt.title(f"Feature Importance - {best_model_name}")
    plt.xlabel("Importance Score")
    plt.show()

# 6️⃣ Pairplot for Feature Relationships
sns.pairplot(df, hue="Failure", palette="coolwarm")
plt.suptitle("Feature Relationships & Failure Distribution", y=1.02)
plt.show()

# -----------------------------------------------------
# 🔹 PREDICTION ON NEW DATA
# -----------------------------------------------------
new_data = pd.DataFrame([[29,0.85,1.6],[33,1.2,1.7],[25,0.7,1.1]], columns=X.columns)
new_data_scaled = scaler.transform(new_data)
prediction = best_model.predict(new_data_scaled)
prediction_prob = best_model.predict_proba(new_data_scaled)[:,1]

for i, row in new_data.iterrows():
    status = "FAIL" if prediction[i]==1 else "RUN SUCCESSFULLY"
    print(f"Input {row.values} → Prediction: {status} (Probability of Failure: {prediction_prob[i]:.2f})")