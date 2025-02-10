import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import ADASYN

import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load dataset
df = pd.read_csv("gendered_talk.csv")
X = df.drop(columns=["label"])
y = df["label"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_scaled)

# Define cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []

for train_index, test_index in kf.split(X_pca, y):
    X_train, X_test = X_pca[train_index], X_pca[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Handle class imbalance
    adasyn = ADASYN()
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
    
    # Reshape for LSTM (samples, timesteps, features)
    X_train_resampled = X_train_resampled.reshape((X_train_resampled.shape[0], X_train_resampled.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Define LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train_resampled.shape[1], 1)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=32, verbose=1)
    
    # Evaluate model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)
    
    print(f'Fold Accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred))

# Print average accuracy
print(f'Average Cross-Validation Accuracy: {np.mean(accuracy_scores):.4f}')

