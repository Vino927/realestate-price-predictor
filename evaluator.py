#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        """
        Evaluates the model performance and prints the evaluation metrics.
        """
        RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
        MSE = mean_squared_error(y_true, y_pred)
        MAE = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"Model Evaluation Metrics:")
        print(f"RMSE (Root Mean Square Error): {RMSE:.2f}")
        print(f"MSE (Mean Squared Error): {MSE:.2f}")
        print(f"MAE (Mean Absolute Error): {MAE:.2f}")
        print(f"R^2 (Coefficient of Determination): {r2:.2f}")

        return RMSE, MSE, MAE, r2

    @staticmethod
    def plot_evaluation(y_true, y_pred):
        """
        Plots the actual vs. predicted values and the distribution of errors.
        """
        # Actual vs. Predicted
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)  # Diagonal line
        plt.title('Actual vs. Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

        # # Error Distribution
        # plt.subplot(1, 2, 2)
        # error = y_pred - y_true
        # plt.hist(error, bins=25, alpha=0.6, color='r')
        # plt.title('Prediction Error Distribution')
        # plt.xlabel('Prediction Error')
        # plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()
