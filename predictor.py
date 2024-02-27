#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.kas.layers import Dense
from tensorflow.keras.callbacks import LearningRateScheduler

class RealEstatePricePredictor:
    def __init__(self, input_dim=5, layers=[100, 100, 100, 200, 200], activation='relu', final_activation='linear', optimizer='adam', clipvalue=0.5, initial_lr=0.001):
        self.input_dim = input_dim
        self.layers = layers
        self.activation = activation
        self.final_activation = final_activation
        self.optimizer = optimizer
        self.clipvalue = clipvalue
        self.initial_lr = initial_lr
        print("Initializing the real estate price predictor model...")
        self.model = self._build_model()

    def _build_model(self):
        print("Building the model...")
        model = Sequential()
        model.add(Dense(self.layers[0], input_dim=self.input_dim, activation=self.activation))
        for layer_size in self.layers[1:]:
            model.add(Dense(layer_size, activation=self.activation))
        model.add(Dense(1, activation=self.final_activation))
        
        if self.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(clipvalue=self.clipvalue, learning_rate=self.initial_lr)
      
        
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def scheduler(self, epoch, lr):
        if epoch < 50:
            return lr
        else:
            adjusted_lr = lr * tf.math.exp(-0.1)
            print(f"Adjusting learning rate to {adjusted_lr:.6f}.")
            return adjusted_lr

    def train(self, X_train, y_train, epochs=100, batch_size=50, validation_split=0.2):
        print("Starting training...")
        lr_schedule = LearningRateScheduler(self.scheduler)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[lr_schedule])
        print("Training completed.")
        return history

    def predict(self, X):
        print("Making predictions...")
        return self.model.predict(X)
