# ==============================================================================
# STEP 0: ENVIRONMENT FIX (Run if you previously encountered import errors)
# ==============================================================================
# !pip install --upgrade scipy numpy scikit-learn pandas
# Note: After running the line above, click "Kernel" -> "Restart Kernel" in Jupyter.

# ==============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ==============================================================================
# STEP 2: LOAD DATASET FROM URL
# ==============================================================================
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# ==============================================================================
# STEP 3: DATA CLEANING
# ==============================================================================
# Remove duplicate rows if any exist
#df = df.drop_duplicates()

# Drop rows where target column 'species' has missing values
df = df.dropna(subset=['species'])

# ==============================================================================
# STEP 4: SEPARATE FEATURES (X) AND TARGET (y)
# ==============================================================================
# Features: sepal_length, sepal_width, petal_length, petal_width
X = df.drop(columns=['species'])

# Target label: species ('setosa', 'versicolor', 'virginica')
y = df['species']

# ==============================================================================
# STEP 5: SPLIT INTO TRAIN AND TEST SETS
# ==============================================================================
# Reserve 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================================================================
# STEP 6: IMPUTATION (FILL MISSING FEATURE VALUES)
# ==============================================================================
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# ==============================================================================
# STEP 7: FEATURE SCALING
# ==============================================================================
# Scale feature features to mean=0 and variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# ==============================================================================
# STEP 8: TRAIN THE KNN MODEL
# ==============================================================================
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# ==============================================================================
# STEP 9: EVALUATE MODEL ACCURACY
# ==============================================================================
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"--- Model Evaluation ---")
print(f"Model Test Accuracy: {accuracy * 100:.2f}%\n")

# ==============================================================================
# STEP 10: PREDICT A CUSTOM UNSEEN DATASET
# ==============================================================================
# Format custom input using original column names
new_flower_data = [[5.1, 3.5, 1.4, 0.2]]
custom_df = pd.DataFrame(new_flower_data, columns=X.columns)

# Apply the same imputation and scaling transformers
custom_imputed = imputer.transform(custom_df)
custom_scaled = scaler.transform(custom_imputed)

# Make prediction
predicted_species = model.predict(custom_scaled)[0]

print(f"--- Custom Prediction ---")
print(f"Input Measurements: {new_flower_data[0]}")
print(f"Predicted Species:  {predicted_species}")