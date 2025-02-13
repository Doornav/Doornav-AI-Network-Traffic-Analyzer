import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def remove_duplicates(self):
        if self.data is not None:
            self.data = self.data.drop_duplicates()

    def fill_missing_values(self):
        if self.data is not None:
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')

    def normalize_features(self):
        if self.data is not None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = (self.data[numeric_cols] - self.data[numeric_cols].mean()) / self.data[numeric_cols].std()
    
    def clean(self):
        self.load_data()
        self.remove_duplicates()
        self.fill_missing_values()
        self.normalize_features()
        return self.data

    def save_clean_data(self, output_path):
        if self.data is not None:
            self.data.to_csv(output_path, index=False)

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "raw_packets.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "clean_packets.csv"
    cleaner = DataCleaner(input_file)
    cleaned_data = cleaner.clean()
    cleaner.save_clean_data(output_file)
    print("Data cleaning completed. Cleaned data saved to", output_file)
