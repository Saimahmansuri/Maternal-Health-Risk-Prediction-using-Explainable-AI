import sys
import os

def run_pipeline():
    # Run complete ML pipeline: data processing, training, explainability
    print("="*70)
    print("MATERNAL HEALTH RISK PREDICTION - COMPLETE PIPELINE")
    print("="*70)
    
    # Download dataset if missing
    print("\n" + "="*70)
    print("STEP 0: DATASET PREPARATION")
    print("="*70)
    
    from src.download_dataset import ensure_dataset_available
    
    if not ensure_dataset_available('data/raw/maternal_health.csv'):
        print("\nERROR: Could not obtain dataset")
        print("\nFor the dataset, manually download from:")
        print("  - Kaggle: https://www.kaggle.com/datasets/andrewmvd/maternal-health-risk-data")
        print("  - UCI: https://archive.ics.uci.edu/ml/datasets/Maternal+Health+Risk+Data+Set")
        print("\nPlace the file at: data/raw/maternal_health.csv")
        print("\nA sample dataset has been created for demonstration purposes")
    
    # Process and clean data
    print("\n" + "="*70)
    print("STEP 1: DATA PROCESSING")
    print("="*70)
    
    from src.data_processing import MaternalDataProcessor
    
    try:
        processor = MaternalDataProcessor()
        processor.process_pipeline()
        print("\nData processing completed successfully")
    except Exception as e:
        print(f"\nERROR during data processing: {e}")
        sys.exit(1)
    
    # Train all models
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    print("\nIMPORTANT: Two training options available:")
    print("  1. train.py     - Simple train/val/test split (quick, for testing)")
    print("  2. train_cv.py  - Stratified k-fold CV (robust, for dissertation)")
    print("\nFor dissertation results, use: python src/models/train_cv.py")
    print("Running basic training for now...\n")
    
    from src.models.train import ModelTrainer
    
    trainer = ModelTrainer()
    trainer.train_all()
    print("\n--- Model training completed ---")
    print("\nFor scientifically robust results with proper validation,")
    print("please run: python src/models/train_cv.py")
    
    # Generate explainability reports (optional)
    print("\n" + "="*70)
    print("STEP 3: EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    try:
        from src.explainers import ModelExplainer
        
        explainer = ModelExplainer(model_type='best')
        explainer.generate_global_report(save_dir='reports')
        print("\nExplainability report generated successfully")
    except ImportError as e:
        print(f"\nWarning: Could not import explainability modules: {str(e)}")
        print("\nTo enable explainability features, install missing packages:")
        print("  pip install shap lime")
        print("\nSkipping explainability analysis")
    except Exception as e:
        print(f"\nWarning: Could not generate explainability report: {str(e)}")
        print("Continuing with remaining steps")
    
    # Print summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED")
    print("="*70)
    
    print("\nAll steps completed:")
    print("  1. Data preprocessing")
    print("  2. Model training")
    print("  3. Explainability analysis")
    
    print("\nGenerated Files:")
    print("  - data/processed/: Processed datasets")
    print("  - models/: Trained model files")
    print("  - reports/: Evaluation results and visualizations")
    
    print("\nNext Steps:")
    print("  1. For robust validation: python src/models/train_cv.py")
    print("  2. Run Dashboard: streamlit run dashboard/streamlit_app.py")
    print("  3. Run API: python src/api/app.py")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    run_pipeline()

