"""
Verification script to check that all components are properly configured
for the improved validation methodology.

Run this before starting the full training pipeline.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    exists = os.path.exists(filepath)
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {description}: {filepath}")
    return exists

def check_imports():
    """Check if required packages are installed."""
    print("\n1. Checking Python Packages...")
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('xgboost', 'XGBoost'),
        ('lightgbm', 'LightGBM'),
        ('imblearn', 'Imbalanced-learn'),
        ('joblib', 'Joblib'),
    ]
    
    all_good = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            all_good = False
    
    return all_good

def check_files():
    """Check if required files exist."""
    print("\n2. Checking Required Files...")
    
    files = [
        ('src/models/train_cv.py', 'Cross-validation training script'),
        ('src/data_processing.py', 'Data processing module'),
        ('run_instructions.txt', 'Reproduction instructions'),
        ('README.md', 'Documentation'),
        ('requirements.txt', 'Dependencies list'),
    ]
    
    all_good = True
    for filepath, description in files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    return all_good

def check_data():
    """Check if data files exist."""
    print("\n3. Checking Data Files...")
    
    raw_data = check_file_exists('data/raw/maternal_health.csv', 'Raw dataset')
    
    processed_files = [
        'data/processed/X_train.npy',
        'data/processed/X_val.npy',
        'data/processed/X_test.npy',
        'data/processed/y_train.npy',
        'data/processed/y_val.npy',
        'data/processed/y_test.npy',
        'data/processed/scaler.pkl',
        'data/processed/feature_names.pkl',
    ]
    
    processed_exist = all(os.path.exists(f) for f in processed_files)
    
    if not raw_data:
        print("\n  Action: Run 'python download_dataset.py' or manually download dataset")
        return False
    
    if not processed_exist:
        print("\n  Action: Run 'python src/data_processing.py' to process data")
        return False
    
    print("  [OK] All data files present")
    return True

def check_random_seeds():
    """Check if random seeds are properly set."""
    print("\n4. Checking Random Seed Configuration...")
    
    files_to_check = [
        'src/data_processing.py',
        'src/models/train.py',
        'src/models/train_cv.py',
    ]
    
    all_good = True
    for filepath in files_to_check:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'RANDOM_SEED = 42' in content:
                    print(f"  [OK] {filepath}")
                else:
                    print(f"  [WARNING] {filepath} - seed not found")
                    all_good = False
        else:
            print(f"  [MISSING] {filepath}")
            all_good = False
    
    return all_good

def check_directories():
    """Check if required directories exist."""
    print("\n5. Checking Directory Structure...")
    
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'reports',
        'src/models',
        'dashboard',
    ]
    
    all_good = True
    for directory in directories:
        if os.path.exists(directory):
            print(f"  [OK] {directory}/")
        else:
            print(f"  [MISSING] {directory}/")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"    Created: {directory}/")
            except Exception as e:
                print(f"    Error creating: {e}")
                all_good = False
    
    return all_good

def main():
    """Run all verification checks."""
    print("="*70)
    print("MATERNAL HEALTH RISK PREDICTION - SETUP VERIFICATION")
    print("="*70)
    
    checks = [
        ("Python Packages", check_imports()),
        ("Required Files", check_files()),
        ("Data Files", check_data()),
        ("Random Seeds", check_random_seeds()),
        ("Directory Structure", check_directories()),
    ]
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("STATUS: All checks passed!")
        print("\nYou are ready to run:")
        print("  python src/models/train_cv.py")
        print("\nThis will generate the dissertation-ready results.")
    else:
        print("STATUS: Some checks failed")
        print("\nPlease address the issues above before running training.")
        print("\nCommon fixes:")
        print("  - Install packages: pip install -r requirements.txt")
        print("  - Download dataset: python download_dataset.py")
        print("  - Process data: python src/data_processing.py")
    
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

