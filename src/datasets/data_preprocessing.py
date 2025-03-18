import os
import pandas as pd

def preprocess_data(input_file, output_file):
    """
    Load raw data from CSV, perform basic cleaning,
    and save the preprocessed data to a new CSV file.
    """
    # Load the raw data
    data = pd.read_csv(input_file)
    
    # Example preprocessing: remove rows with missing values
    data = data.dropna()
    
    # Additional preprocessing steps can be added here...
    # e.g., data normalization, feature engineering, categorical encoding, etc.
    
    # Save the preprocessed data
    data.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    # Define paths for input raw data and output cleaned data
    input_path = "datasets/raw_data.csv"    # Replace with your actual raw data file
    output_path = "datasets/data.csv"
    
    # Ensure the datasets folder exists
    os.makedirs("datasets", exist_ok=True)
    
    preprocess_data(input_path, output_path)