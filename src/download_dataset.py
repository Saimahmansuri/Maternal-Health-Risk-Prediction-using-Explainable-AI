import os
import urllib.request
import pandas as pd
import ssl
from pathlib import Path


def download_dataset(output_path='data/raw/maternal_health.csv', force=False):
    # Download dataset from UCI or create sample if download fails
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip download if file already exists
    if os.path.exists(output_path) and not force:
        print(f"✓ Dataset already exists at: {output_path}")
        return True
    
    print("="*70)
    print("DOWNLOADING MATERNAL HEALTH RISK DATASET")
    print("="*70)
    
    # Try multiple download sources
    dataset_urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv",
        "https://raw.githubusercontent.com/andrewmvd/maternal-health-risk/master/Maternal%20Health%20Risk%20Data%20Set.csv",
    ]
    
    def create_sample_dataset():
        # Fallback: create synthetic dataset if download fails
        print("\n⚠ Could not download dataset from online sources.")
        print("Creating a sample dataset for demonstration...")
        
        import numpy as np
        
        np.random.seed(42)
        n_samples = 1014
        
        # Generate synthetic patient data
        data = {
            'Age': np.random.randint(10, 70, n_samples),
            'SystolicBP': np.random.randint(70, 160, n_samples),
            'DiastolicBP': np.random.randint(49, 100, n_samples),
            'BS': np.random.uniform(6.0, 19.0, n_samples).round(1),
            'BodyTemp': np.random.uniform(98.0, 103.0, n_samples).round(1),
            'HeartRate': np.random.randint(7, 90, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Assign risk levels based on feature values
        def assign_risk(row):
            risk_score = 0
            if row['Age'] > 35: risk_score += 1
            if row['SystolicBP'] > 140: risk_score += 2
            if row['DiastolicBP'] > 90: risk_score += 2
            if row['BS'] > 10: risk_score += 1
            if row['BodyTemp'] > 100: risk_score += 1
            if row['HeartRate'] > 80: risk_score += 1
            
            if risk_score >= 5:
                return 'high risk'
            elif risk_score >= 2:
                return 'mid risk'
            else:
                return 'low risk'
        
        df['RiskLevel'] = df.apply(assign_risk, axis=1)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Sample dataset created at: {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Note: This is synthetic data for demonstration purposes.")
        print(f"\n  For real dataset, download manually from:")
        print(f"  - Kaggle: https://www.kaggle.com/datasets/andrewmvd/maternal-health-risk-data")
        print(f"  - UCI: https://archive.ics.uci.edu/ml/datasets/Maternal+Health+Risk+Data+Set")
        
        return True
    
    # Try each download URL
    for i, url in enumerate(dataset_urls):
        try:
            print(f"\nAttempt {i+1}/{len(dataset_urls)}: Downloading from {url[:60]}...")
            
            # Disable SSL verification for older repositories
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Show download progress
            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    print(f"\rProgress: {percent}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
            print()
            
            # Validate downloaded file
            try:
                df = pd.read_csv(output_path)
                expected_columns = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'RiskLevel']
                
                if all(col in df.columns for col in expected_columns):
                    print(f"✓ Dataset downloaded successfully!")
                    print(f"  Location: {output_path}")
                    print(f"  Shape: {df.shape}")
                    print(f"  Columns: {list(df.columns)}")
                    return True
                else:
                    print("⚠ Downloaded file has unexpected format. Trying next source...")
                    os.remove(output_path)
                    continue
                    
            except Exception as e:
                print(f"⚠ Error validating downloaded file: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                continue
                
        except Exception as e:
            print(f"⚠ Download failed: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            continue
    
    # Create sample dataset if all downloads failed
    print("\n" + "="*70)
    return create_sample_dataset()


def check_dataset_exists(path='data/raw/maternal_health.csv'):
    # Check if dataset file exists and has correct format
    if not os.path.exists(path):
        return False
    
    try:
        df = pd.read_csv(path)
        expected_columns = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'RiskLevel']
        return all(col in df.columns for col in expected_columns)
    except:
        return False


def ensure_dataset_available(path='data/raw/maternal_health.csv'):
    # Check if dataset exists, download if missing
    if check_dataset_exists(path):
        print(f"✓ Dataset found at: {path}")
        return True
    
    print(f"Dataset not found. Downloading automatically...")
    return download_dataset(path)


if __name__ == "__main__":
    import sys
    
    # Handle command line arguments
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/maternal_health.csv'
    force_download = '--force' in sys.argv
    
    success = download_dataset(output_path, force=force_download)
    
    if success:
        print("\n" + "="*70)
        print("DATASET READY!")
        print("="*70)
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("DATASET DOWNLOAD FAILED!")
        print("="*70)
        sys.exit(1)

