import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import ADASYN
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------- STEP 1: LOAD DATASET -----------------
print("\n✅ Starting LSTM training script...")
try:
    df = pd.read_csv("gendered_talk.csv")  # Ensure correct dataset
    print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print("❌ ERROR: Dataset not found! Make sure 'gendered_talk.csv' is in the script directory.")
    exit()

# ----------------- STEP 2: PREPARE DATA -----------------
X = df.drop(columns=["label"])  # Feature set
y = df["label"]  # Target variable (0 = Female, 1 = Male)

# Standardize Features (Important for LSTM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------- STEP 3: REDUCE FEATURES USING PCA -----------------
print("\n🔹 Reducing features using PCA...")
pca = PCA(n_components=10)  # Keep only the top 10 features
X_pca = pca.fit_transform(X_scaled)
print(f"✅ Reduced feature set: {X_pca.shape[1]} components retained")

# ----------------- STEP 4: ADD NOISE TO TEST GENERALIZATION -----------------
noise_factor = 0.01  # Small noise to prevent overfitting
X_noisy = X_pca + noise_factor * np.random.normal(size=X_pca.shape)
print("✅ Added noise to dataset to simulate real-world conditions")

# ----------------- STEP 5: HANDLE CLASS IMBALANCE (ADASYN) -----------------
print("\n🔹 Balancing dataset using ADASYN...")
adasyn = ADASYN(sampling_strategy=0.8, random_state=42)  # Alternative to SMOTE
X_balanced, y_balanced = adasyn.fit_resample(X_noisy, y)  # Resample dataset
print(f"✅ After ADASYN: {sum(y_balanced == 0)} Female, {sum(y_balanced == 1)} Male")

# Reshape input for LSTM (samples, time steps, features)
X_lstm = X_balanced.reshape(X_balanced.shape[0], 1, X_balanced.shape[1])

# ----------------- STEP 6: SPLIT DATA INTO TRAIN/TEST (NEW TEST SET) -----------------
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_balanced, test_size=0.2, random_state=202, stratify=y_balanced)
print(f"✅ Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ----------------- STEP 7: BUILD LSTM MODEL -----------------
print("\n🔹 Building LSTM Model...")
model = Sequential([
    LSTM(16, return_sequences=True, input_shape=(1, X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.5),
    LSTM(8),
    BatchNormalization(),
    Dropout(0.5),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")  # Output layer (Binary classification)
])

# Compile Model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print("✅ LSTM Model Built Successfully!")

# ----------------- STEP 8: TRAIN MODEL -----------------
print("\n🔹 Training LSTM Model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, verbose=1)

# ----------------- STEP 9: EVALUATE MODEL -----------------
print("\n🔹 Evaluating LSTM Model...")
y_pred_lstm = (model.predict(X_test) > 0.5).astype("int32")

accuracy = accuracy_score(y_test, y_pred_lstm)
print(f"📌 Accuracy (LSTM Neural Network): {accuracy:.4f}")

print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred_lstm))

# ----------------- STEP 10: CHECK PREDICTION DISTRIBUTION -----------------
unique, counts = np.unique(y_pred_lstm, return_counts=True)
print(f"\n📌 Predictions distribution: {dict(zip(unique, counts))}")

# If the model is predicting only one class, warn the user
if len(unique) == 1:
    print("⚠ WARNING: Model is predicting only one class. Consider adjusting model architecture.")

# ----------------- STEP 11: TRAINING LOSS & ACCURACY PLOTS -----------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("LSTM Training & Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("LSTM Training & Validation Accuracy")
plt.legend()

plt.show()

print("\n✅ LSTM classification script execution complete!")

try:
    model.save("gender_recognition_lstm.h5")
    print("✅ Model successfully saved as gender_recognition_lstm.h5")
except Exception as e:
    print(f"❌ Error saving model: {e}")

