from src.download_dataset import download_dataset
import sys

if __name__ == "__main__":
    # Standalone script to download dataset
    print("="*70)
    print("MATERNAL HEALTH RISK DATASET - AUTOMATIC DOWNLOADER")
    print("="*70)
    
    # Get output path from command line or use default
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/maternal_health.csv'
    force_download = '--force' in sys.argv or '-f' in sys.argv
    
    if force_download:
        print("Force download enabled. Will overwrite existing file.")
    
    # Download dataset
    success = download_dataset(output_path, force=force_download)
    
    # Show results
    if success:
        print("\n" + "="*70)
        print("✓ DATASET READY!")
        print("="*70)
        print(f"\nYou can now run: python run_pipeline.py")
    else:
        print("\n" + "="*70)
        print("⚠ DOWNLOAD ISSUE")
        print("="*70)
        print("\nA sample dataset has been created for demonstration.")
        print("For the real dataset, download manually from:")
        print("  - Kaggle: https://www.kaggle.com/datasets/andrewmvd/maternal-health-risk-data")
        print("  - UCI: https://archive.ics.uci.edu/ml/datasets/Maternal+Health+Risk+Data+Set")

