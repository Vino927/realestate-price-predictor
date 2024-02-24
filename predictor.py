from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf

class RealEstatePricePredictor:
    def __init__(self):
        # Initialize the model upon creation of an instance of RealEstatePricePredictor
        print("Initializing the real estate price predictor model...")
        self.model = self._build_model()

    def _build_model(self):
        # Private method to build a Sequential neural network model
        print("Building the model...")
        model = Sequential([
            Dense(100, input_dim=5, activation='relu'),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(200, activation='relu'),
            Dense(200, activation='relu'),
            Dense(1, activation='linear')
        ])

        # Use Adam optimizer with gradient clipping
        optimizer = tf.keras.optimizers.Adam(clipvalue=0.5)

        # Compile the model
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def scheduler(self, epoch, lr):
        # Learning rate scheduler function to adjust the learning rate over epochs
        if epoch < 50:
            return lr  # No change in learning rate for the first 50 epochs
        else:
            adjusted_lr = lr * tf.math.exp(-0.1)  # Exponentially decay the learning rate
            print(f"Adjusting learning rate to {adjusted_lr:.6f}.")
            return adjusted_lr

    def train(self, X_train, y_train):
        # Train the model on the training data
        print("Starting training...")
        # Apply learning rate scheduler
        lr_schedule = LearningRateScheduler(self.scheduler)
        history = self.model.fit(X_train, y_train, epochs=100, batch_size=50, validation_split=0.2, callbacks=[lr_schedule])
        print("Training completed.")
        return history

    def predict(self, X):
        # Predict the target values for a given input
        print("Making predictions...")
        return self.model.predict(X)