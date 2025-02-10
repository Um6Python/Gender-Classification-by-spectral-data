#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

def main():
    # Load the dataset from 'gendered_talk.csv'
    data = pd.read_csv("gendered_talk.csv")
    print("Columns in the dataset:", data.columns.tolist())

    # Use "label" as the target variable.
    X = data.drop("label", axis=1)
    y = data["label"]

    # Split the data into training and testing sets (80% train, 20% test).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a pipeline that scales the data and then fits a logistic regression model.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(solver="lbfgs", max_iter=20000))
    ])

    # Train the model.
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set.
    y_pred = pipeline.predict(X_test)

    # Evaluate and print the model's performance on the test set.
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Test F1 Score:", f1_score(y_test, y_pred))
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_pred))

    # Evaluate and print the model's performance on the training set.
    y_train_pred = pipeline.predict(X_train)
    print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Training F1 Score:", f1_score(y_train, y_train_pred))
    print("Classification Report (Training Set):")
    print(classification_report(y_train, y_train_pred))

    # Use cross-validation to further assess model generalization.
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print("Cross-Validation Accuracy Scores:", cv_scores)
    print("Mean Cross-Validation Accuracy:", cv_scores.mean())

if __name__ == "__main__":
    main()

