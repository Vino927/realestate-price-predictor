import pandas as pd
import os


class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        print("Attempting to load data...")
        if not os.path.exists(self.filepath):
            print(f"Error: The file '{self.filepath}' does not exist.")
            return
        try:
            self.df = pd.read_csv(self.filepath)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")

    def preprocess(self):
        # Fill missing values for specified columns with their mode
        print("filling empty cells and removing outliers...")

        for column in ['bed', 'bath', 'acre_lot', 'house_size']:
            self.df[column].fillna(self.df[column].mode()[0], inplace=True)

        # Drop rows with missing values in 'zip_code', 'city', and 'price' columns
        self.df = self.df.dropna(subset=['zip_code', 'city', 'price'])

        # Remove top and bottom 25 percentile outliers
        cols = ['bed', 'bath', 'acre_lot', 'house_size', 'price']
        Q1 = self.df[cols].quantile(0.25)
        Q3 = self.df[cols].quantile(0.75)
        IQR = Q3 - Q1
        self.df = self.df[~((self.df[cols] < (Q1 - 1.5 * IQR)) | (self.df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    def get_cleaned_data(self):
        return self.df

