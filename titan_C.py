import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# ==============================================================================
# 1. تحميل البيانات
# ==============================================================================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ------------------------------------------------------------------------------
# طباعة البيانات قبل المعالجة (Before Preprocessing)
# ------------------------------------------------------------------------------
print("=== 1. 1 - 5 lines of the dataset before a preprocessing) ===")
print(df.head())

print("\n=== 2. Before Preprocessing  ===") # معلومات ونوع البيانات وحجم الفراغات (قبل المعالجة
print(df.info())

print("\n=== 3. Before Preprocessing ===") # عدد القيم المفقودة في كل عمود (قبل المعالجة)
print(df.isnull().sum())

# ==============================================================================
# 2.  (Preprocessing)
# ==============================================================================
# أ) حذف الأعمدة غير المؤثرة
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# ب) التعويض عن القيم المفقودة
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# ج) تحويل النصوص إلى أرقام (One-Hot Encoding)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# د) فصل الميزات والهدف
X = df.drop(columns=['Survived'])
y = df['Survived']

# هـ) تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# و) تحجيم القيم العددية (Standard Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------------------
#  (After Preprocessing)
# ------------------------------------------------------------------------------
# تحويل مصفوفة التدريب المحجمة إلى DataFrame لعرضها كجدول
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)

print("\n=== 4.  (Scaled Features) ===") # البيانات بعد التحويل، الترميز الرقمي، والتحجيم
print(X_train_scaled_df.head())

print("\n=== 5. After Preprocessing ===")
print(X_train_scaled_df.isnull().sum())

# ==============================================================================
# 3. Training
# ==============================================================================
model = RandomForestClassifier(n_estimators=101, random_state=42)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
print(f"\n Trained Model Acc. : {accuracy_score(y_test, predictions) * 100:.2f}%\n")

# ==============================================================================
# 4.  (Model Plots)
# ==============================================================================

# (Feature Importance)
plt.figure(figsize=(10, 5))
importances = model.feature_importances_
indices = np.argsort(importances)

plt.title(" (Feature Importance Chart)")
plt.barh(range(len(indices)), importances[indices], color='skyblue', align="center")
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel( "نسبة التأثير على القرار")
plt.tight_layout()
plt.show()

#  (Confusion Matrix)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, cmap='Blues', ax=ax)
plt.title("مخطط مصفوفة الارتباك (Confusion Matrix)")
plt.show()