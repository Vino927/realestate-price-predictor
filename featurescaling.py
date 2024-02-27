#!/usr/bin/env python3
from sklearn.preprocessing import MinMaxScaler

class FeatureScaler:

    def __init__(self):
        # Initialize MinMaxScaler objects for scaling features and the target variable independently
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        print("FeatureScaler initialized with MinMaxScaler for both features and target.")

    def fit_transform(self, X, y):
        # Scale the features and the target variable
        print("Fitting and transforming features and target variable.")
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1,1))
        print("Features and target variable scaled successfully.")
        return X_scaled, y_scaled

    def inverse_transform(self, y):
        # Inverse transform the scaled target variable back to its original scale
        print("Inverse transforming the target variable.")
        return self.target_scaler.inverse_transform(y)

    def transform(self, X):
        # Transform the features using the existing scaler fit
        print("Transforming features using the existing scaler fit.")
        return self.feature_scaler.transform(X)
