
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

# Initialize the predictor, train it, and make predictions on the test set
predictor = RealEstatePricePredictor()
predictor.train(X_train, y_train)

y_pred_scaled = predictor.predict(X_test)
y_pred = feature_scaler.inverse_transform(y_pred_scaled)
y_test_inv = feature_scaler.inverse_transform(y_test)

# Evaluate the model's performance and plot the evaluation
rmse, mse, mae, r2 = ModelEvaluator.evaluate(y_test_inv.flatten(), y_pred.flatten())  # Flatten arrays if necessary
ModelEvaluator.plot_evaluation(y_test_inv.flatten(), y_pred.flatten()) 