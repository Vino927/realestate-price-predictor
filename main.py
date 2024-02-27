#!/usr/bin/env python3
from dataloader import DataPreprocessor
from featurescaling import FeatureScaler
from predictor import RealEstatePricePredictor
from evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split

data_preprocessor = DataPreprocessor('realtor_ma_only.csv')
data_preprocessor.load_data()
data_preprocessor.preprocess()
df = data_preprocessor.get_cleaned_data()

selected_features = ['bed', 'bath', 'acre_lot', 'zip_code', 'house_size']
X = df[selected_features]
y = df['price']

# Initialize and fit-transform features and target using the feature scaler
feature_scaler = FeatureScaler()
X_scaled, y_scaled = feature_scaler.fit_transform(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25)

# Initialize the predictor with input parameters for the building the model
predictor = RealEstatePricePredictor(
    input_dim=len(selected_features),  # Number of features used as input to the model.
    layers=[100, 100, 100, 200, 200],  # Architecture: Number of neurons in each layer of the network.
    activation='relu',  # Activation function for intermediate layers: ReLU (Rectified Linear Unit).
    final_activation='linear',  # Activation function for the output layer: linear for regression tasks.
    optimizer='adam',  # Optimization algorithm
    clipvalue=0.5,  # Gradient clipping value to prevent exploding gradients.
    initial_lr=0.001  # Initial learning rate for the Adam optimizer.
)


# Train the predictor with additional parameters for training
predictor.train(
    X_train, 
    y_train,
    epochs=100,  # Number of epochs
    batch_size=50,  # Batch size
    validation_split=0.2  # Fraction of data to use as validation
)

y_pred_scaled = predictor.predict(X_test)  # Predict the scaled target values using the test features.
y_pred = feature_scaler.inverse_transform(y_pred_scaled)  # Inverse transform the predictions to original scale.
y_test_inv = feature_scaler.inverse_transform(y_test)  # Inverse transform the true test target values to original scale.
 

# Evaluate the model's performance and plot the evaluation
rmse, mse, mae, r2 = ModelEvaluator.evaluate(y_test_inv.flatten(), y_pred.flatten())  # Flatten arrays if necessary
ModelEvaluator.plot_evaluation(y_test_inv.flatten(), y_pred.flatten()) 