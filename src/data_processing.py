import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class MaternalDataProcessor:
    
    def __init__(self, raw_data_path='data/raw/maternal_health.csv',
                 processed_data_path='data/processed/maternal_processed.csv'):
        # Initialize processor with paths and scalers
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_data(self):
        # Load raw dataset, try to download if missing
        print(f"Loading data from {self.raw_data_path}...")
        
        # Check if dataset exists, try to download if not
        if not os.path.exists(self.raw_data_path):
            print(f"Dataset not found at {self.raw_data_path}")
            print("Attempting automatic download...")
            
            try:
                from download_dataset import ensure_dataset_available
                if not ensure_dataset_available(self.raw_data_path):
                    raise FileNotFoundError("Failed to download dataset automatically")
            except ImportError:
                print("Error: Could not import download module")
                print("Please download the dataset manually from:")
                print("  - Kaggle: https://www.kaggle.com/datasets/andrewmvd/maternal-health-risk-data")
                print("  - UCI: https://archive.ics.uci.edu/ml/datasets/Maternal+Health+Risk+Data+Set")
                raise FileNotFoundError(f"File not found at {self.raw_data_path}")
        
        try:
            df = pd.read_csv(self.raw_data_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        
    def check_data_quality(self, df):
        # Check data quality: missing values, duplicates, types
        print("\n=== Data Quality Check ===")
        report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'dtypes': df.dtypes.to_dict()
        }
        
        print(f"Dataset shape: {report['shape']}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        print(f"Number of duplicate rows: {report['duplicates']}")
        print(f"\nData types:\n{df.dtypes}")
        
        return report
        
    def clean_data(self, df):
        # Clean data: remove duplicates and handle missing values
        print("\n=== Cleaning Data ===")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        print(f"Removed {removed_duplicates} duplicate rows")
        
        # Fill missing values with median for numerical columns
        if df.isnull().sum().sum() > 0:
            print("Handling missing values...")
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"Filled {col} missing values with median: {median_val}")
        else:
            print("No missing values found")
            
        return df

    def engineer_features(self, df):
        # Add new clinical features
        print("\n=== Feature Engineering ===")
        
        # 1. Pulse Pressure (SystolicBP - DiastolicBP)
        df['PulsePressure'] = df['SystolicBP'] - df['DiastolicBP']
        print("Feature created: PulsePressure")
        
        # 2. BodyTemp in Celsius (some models prefer standard units)
        df['BodyTemp_C'] = (df['BodyTemp'] - 32) * 5/9
        print("Feature created: BodyTemp_C")
        
        return df
        
    def encode_target(self, df, target_column='RiskLevel'):
        # Encode risk level: Low=0, Mid=1, High=2
        print("\n=== Encoding Target Variable ===")
        
        risk_mapping = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
        
        # Normalize target values to lowercase and map
        df[target_column] = df[target_column].str.lower()
        df[f'{target_column}_Encoded'] = df[target_column].map(risk_mapping)
        
        print(f"Target encoding mapping: {risk_mapping}")
        print(f"Target distribution:\n{df[f'{target_column}_Encoded'].value_counts().sort_index()}")
        
        return df
        
    def scale_features(self, X_train, X_val, X_test):
        # Scale features using StandardScaler (fit on train, transform all)
        print("\n=== Scaling Features ===")
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Features scaled using StandardScaler")
        print(f"Feature means: {self.scaler.mean_}")
        print(f"Feature stds: {self.scaler.scale_}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
        
    def split_data(self, df, target_column='RiskLevel_Encoded', 
                   train_size=0.70, val_size=0.15, test_size=0.15, 
                   random_state=RANDOM_SEED):
        # Split data into train/val/test with stratification (70/15/15)
        # Uses fixed random seed for reproducibility
        print("\n=== Splitting Data ===")
        print(f"Random seed: {random_state}")
        
        # Check split proportions
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Train, val, and test sizes must sum to 1.0"
        
        # Separate features and target
        feature_cols = [col for col in df.columns 
                       if col not in ['RiskLevel', 'RiskLevel_Encoded']]
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df[target_column]
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size), 
            random_state=random_state, stratify=y
        )
        
        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio),
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"\nFeatures used: {feature_cols}")
        
        # Show class distribution in each split
        print("\nClass distribution:")
        print(f"Train: {y_train.value_counts().sort_index().to_dict()}")
        print(f"Val: {y_val.value_counts().sort_index().to_dict()}")
        print(f"Test: {y_test.value_counts().sort_index().to_dict()}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def save_processed_data(self, X_train, X_val, X_test, 
                           y_train, y_val, y_test):
        # Save processed data and preprocessing objects to disk
        print("\n=== Saving Processed Data ===")
        
        os.makedirs('data/processed', exist_ok=True)
        
        # Save data as numpy arrays
        np.save('data/processed/X_train.npy', X_train)
        np.save('data/processed/X_val.npy', X_val)
        np.save('data/processed/X_test.npy', X_test)
        np.save('data/processed/y_train.npy', y_train)
        np.save('data/processed/y_val.npy', y_val)
        np.save('data/processed/y_test.npy', y_test)
        
        # Save scaler and feature names for later use
        joblib.dump(self.scaler, 'data/processed/scaler.pkl')
        joblib.dump(self.feature_columns, 'data/processed/feature_names.pkl')
        
        print("Processed data saved to data/processed/")
        print("- X_train.npy, X_val.npy, X_test.npy")
        print("- y_train.npy, y_val.npy, y_test.npy")
        print("- scaler.pkl")
        print("- feature_names.pkl")
        
    def process_pipeline(self):
        # Run complete data processing pipeline
        print("="*60)
        print("MATERNAL HEALTH DATA PROCESSING PIPELINE")
        print("="*60)
        
        df = self.load_data()
        self.check_data_quality(df)
        df = self.clean_data(df)
        df = self.engineer_features(df)
        df = self.encode_target(df)
        
        # Split into train/val/test sets
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )
        
        # Save everything to disk
        self.save_processed_data(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test
        )
        
        print("\n" + "="*60)
        print("DATA PROCESSING COMPLETE!")
        print("="*60)
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test)


def load_processed_data():
    # Load previously processed data and preprocessing objects
    print("Loading processed data...")
    
    X_train = np.load('data/processed/X_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    scaler = joblib.load('data/processed/scaler.pkl')
    feature_names = joblib.load('data/processed/feature_names.pkl')
    
    print("Data loaded successfully!")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names


if __name__ == "__main__":
    processor = MaternalDataProcessor()
    processor.process_pipeline()

