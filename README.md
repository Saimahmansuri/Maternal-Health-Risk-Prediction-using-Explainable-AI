# Maternal Health Risk Prediction System

A machine learning-based decision support system for assessing maternal health risk levels during pregnancy. This is a research prototype designed for pilot validation and academic evaluation.

## Important Notice

**This is a research prototype and NOT a clinical diagnostic tool.** The system requires prospective validation before any clinical consideration. All predictions must be verified by qualified healthcare professionals.

## Project Overview

This project implements a supervised machine learning system to classify maternal health risk into three categories (Low, Mid, High) based on six clinical measurements. The system is designed to support healthcare providers in identifying high-risk pregnancies for early intervention, while maintaining transparent and explainable predictions.

The primary contribution of this work is the implementation of robust validation methodology that addresses common pitfalls in medical ML research, including data leakage prevention, proper handling of class imbalance, and comprehensive cross-validation.

## Key Features

- Stratified k-fold cross-validation to ensure robust performance estimates
- Proper SMOTE handling within training folds to prevent data leakage
- Comparison of different class imbalance strategies
- Explainable AI using SHAP and LIME
- Web dashboard for interactive risk assessment
- REST API for system integration
- Comprehensive documentation for reproducibility

## Dataset

**Source:** UCI Machine Learning Repository / Kaggle  
**Name:** Maternal Health Risk Data Set  
**Original Study:** Ahmed et al.  
**Sample Size:** Approximately 1000 pregnant women  
**Geographic Region:** Bangladesh (rural and urban areas)  
**Collection Method:** IoT-based risk monitoring system

### Features

The dataset includes six clinical measurements:

1. **Age** - Maternal age in years
2. **SystolicBP** - Systolic blood pressure (mmHg)
3. **DiastolicBP** - Diastolic blood pressure (mmHg)
4. **BS** - Blood sugar level (mmol/L)
5. **BodyTemp** - Body temperature (Fahrenheit)
6. **HeartRate** - Heart rate (beats per minute)

### Target Variable

**RiskLevel:** Three-class ordinal variable
- Low Risk (0)
- Mid Risk (1)
- High Risk (2)

### Dataset Limitations

1. **Limited sample size** (n approximately 1000) restricts generalization capabilities
2. **Single geographic region** limits applicability to other populations
3. **No temporal data** - single time-point measurements without pregnancy progression tracking
4. **Limited feature set** - only six measurements, missing many clinically relevant factors
5. **Potential selection bias** in data collection methodology
6. **Class imbalance** with fewer high-risk cases than low-risk cases
7. **No validation of measurement accuracy** or inter-rater reliability

## Installation

### Prerequisites

- Python 3.10 or higher
- 2GB free disk space
- Virtual environment (recommended)

### Setup Instructions

1. Clone or download the project repository

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. If version conflicts occur:

```bash
python fix_dependencies.py
```

5. Verify installation:

```bash
python check_env.py
```

## Reproducibility

All experiments use fixed random seeds for reproducibility:

- **Global seed:** 42
- **Set in:** src/data_processing.py, src/models/train.py, src/models/train_cv.py
- **Controls:** numpy, random, tensorflow, sklearn operations

For complete reproduction instructions, see `run_instructions.txt`.

## Usage

### Data Processing

Process the raw dataset with stratified splitting and feature engineering:

```bash
python src/data_processing.py
```

This performs:
- Data cleaning (duplicate removal, missing value handling)
- Feature engineering (PulsePressure, BodyTemp_C)
- Stratified train-validation-test split (70/15/15)
- Feature scaling (StandardScaler, fitted only on training data)

### Model Training with Cross-Validation

**For dissertation and publication**, use the robust cross-validation training:

```bash
python src/models/train_cv.py
```

This implements:
- Stratified 5-fold cross-validation
- SMOTE applied only within training folds (no data leakage)
- Comparison of resampling strategies
- Evaluation of top models (LightGBM, XGBoost, Gradient Boosting)
- Mean and standard deviation reporting for all metrics
- Model selection based on High-Risk recall (clinical priority)

Outputs:
- `reports/cv_results_summary.csv` - Mean ± std for each model
- `reports/cv_results_folds.csv` - Raw fold-by-fold results
- `reports/resampling_strategy_comparison.csv` - Strategy comparison
- `models/best_model.pkl` - Final trained model
- `models/model_metadata.json` - Model configuration and performance

### Web Dashboard

Launch the interactive web interface:

```bash
streamlit run dashboard/streamlit_app.py
```

Access at: http://localhost:8501

The dashboard provides:
- Single patient risk assessment
- Probability distributions across risk levels
- LIME feature contribution analysis
- Batch prediction from CSV files
- Clinical decision support recommendations

### REST API

Launch the API server:

```bash
python src/api/app.py
```

Access documentation at: http://localhost:8000/docs

## Validation Methodology

This project implements rigorous validation to ensure results are not optimistic:

### Stratified K-Fold Cross-Validation

- **Folds:** 5
- **Strategy:** Stratified to maintain class distribution
- **Process:** Each sample validated exactly once
- **Reporting:** Mean ± standard deviation for all metrics

### Data Leakage Prevention

- SMOTE applied only within training folds using imblearn Pipeline
- Validation data never exposed to synthetic samples
- Test set completely held out from all training processes
- Scaler fitted only on training data

### Class Imbalance Handling

Three strategies compared:
1. No resampling (baseline)
2. SMOTE oversampling (applied within CV folds only)
3. Class weighting (native model parameter)

### Model Selection Criteria

Primary criterion: **High-Risk recall**
- Rationale: Clinical priority to minimize false negatives in high-risk cases
- Balanced with overall F1 score and accuracy
- Selection based on cross-validation performance, not test set

### Test Set Evaluation

- Held-out test set (15% of data)
- Used only for final unbiased performance estimate
- Never used for model selection or hyperparameter tuning

## Performance

Based on stratified 5-fold cross-validation (results will vary slightly due to random initialization):

### Expected Performance Ranges

**Top Models:** LightGBM, XGBoost, Gradient Boosting

Metrics (mean ± std from 5-fold CV):
- Accuracy: 0.85-0.95 ± 0.02-0.05
- F1 Score (weighted): 0.85-0.95 ± 0.02-0.05
- F1 Score (macro): 0.80-0.90 ± 0.03-0.06
- High-Risk Recall: 0.70-0.90 ± 0.05-0.15

Note: Standard deviations indicate performance variability across folds. Higher variability suggests sensitivity to specific data splits.

### Confidence Intervals

For 95% confidence intervals on metrics, use:
CI = mean ± 1.96 × (std / sqrt(n_folds))

### Statistical Comparison

When comparing models, consider:
- Overlapping confidence intervals indicate no significant difference
- Clinical significance may differ from statistical significance
- High-Risk recall takes priority over overall accuracy

## Explainability

The system provides interpretable predictions through:

### SHAP (SHapley Additive exPlanations)

- Global feature importance across all predictions
- Individual prediction explanations with waterfall plots
- Shows contribution of each feature to risk assessment

### LIME (Local Interpretable Model-agnostic Explanations)

- Local explanations for individual patients
- Feature contributions in raw measurement units
- Identifies factors increasing or decreasing risk

### Important Notes on Explainability

1. **Association vs Causation:** Explanations show statistical associations, not causal relationships
2. **Approximations:** SHAP and LIME provide approximations of model behavior
3. **Limitations:** Explanations may not capture complex feature interactions
4. **Clinical Integration:** Explanations should support, not replace, clinical judgment

## Limitations and Considerations

### Dataset Limitations

1. **Small sample size** (approximately 1000) limits statistical power and generalization
2. **Single geographic region** - trained on Bangladesh population, may not generalize
3. **Lack of longitudinal data** - no tracking of pregnancy progression
4. **Missing clinical factors** - many relevant measurements not included
5. **Potential bias** in data collection and labeling process

### Model Limitations

1. **Not validated prospectively** - requires real-world validation studies
2. **No bias evaluation** across different demographic groups
3. **No temporal validation** - performance over time unknown
4. **Limited to six features** - does not consider medical history, lab results, imaging
5. **Class imbalance** may affect minority class predictions despite mitigation strategies

### Methodological Limitations

1. **Hyperparameters not exhaustively tuned** - focused on default or simple configurations
2. **Limited model comparison** - focused on top 3 models for efficiency
3. **Single dataset** - no external validation dataset
4. **Synthetic data concerns** - SMOTE creates artificial samples that may not reflect real variability

### Deployment Considerations

1. **Not clinically validated** - requires regulatory approval before clinical use
2. **No integration with EHR systems** - standalone prototype only
3. **No monitoring framework** - performance degradation detection not implemented
4. **No audit trail** - limited logging for clinical governance
5. **Privacy not fully addressed** - requires HIPAA/GDPR compliance assessment

## Ethical and Regulatory Considerations

### Ethical Concerns

- Model trained on specific population may perpetuate existing biases
- False negatives in high-risk detection could lead to missed interventions
- Over-reliance on automated predictions may reduce clinical judgment
- Informed consent and patient autonomy must be maintained
- Transparency about model limitations is essential

### Regulatory Requirements

Before clinical deployment, this system would require:
- Clinical validation studies with prospective data
- Regulatory approval (FDA, CE marking, etc.)
- Bias and fairness evaluation across demographic groups
- Privacy impact assessment (HIPAA, GDPR compliance)
- Clinical governance framework
- Regular performance monitoring and recalibration
- Clear protocols for human oversight

## Future Work

### Recommended Improvements

1. **Data Collection**
   - Larger, more diverse dataset from multiple regions
   - Longitudinal tracking throughout pregnancy
   - Additional clinical features (lab results, medical history, ultrasound)
   - Validation dataset from different healthcare settings

2. **Methodology**
   - External validation on independent dataset
   - Prospective validation study in clinical setting
   - Bias evaluation across demographic subgroups
   - Temporal validation to assess model degradation
   - Comparison with clinical risk scores (as baseline)

3. **Model Enhancement**
   - Time-series analysis for pregnancy progression
   - Ensemble methods combining multiple models
   - Uncertainty quantification for predictions
   - Calibration analysis for probability estimates

4. **Clinical Integration**
   - Integration with electronic health records
   - Clinical decision support workflow design
   - Usability testing with healthcare providers
   - Patient-facing interface for risk communication

5. **Deployment**
   - Monitoring framework for model performance
   - Feedback loop for continuous improvement
   - Audit trail for clinical governance
   - Privacy-preserving deployment architecture

## Project Structure

```
maternal-risk-project/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Processed data (generated)
├── models/                     # Trained models (generated)
├── reports/                    # Results and visualizations (generated)
├── notebooks/                  # Jupyter notebooks for exploration
├── src/
│   ├── data_processing.py      # Data preprocessing pipeline
│   ├── models/
│   │   ├── train.py            # Basic training (single split)
│   │   ├── train_cv.py         # Cross-validation training (use this)
│   │   └── predict.py          # Prediction service
│   ├── explainers.py           # SHAP and LIME explanations
│   └── api/
│       └── app.py              # REST API
├── dashboard/
│   └── streamlit_app.py        # Web dashboard
├── requirements.txt            # Dependencies with versions
├── run_instructions.txt        # Detailed reproduction instructions
└── README.md                   # This file
```

## Dependencies

Key dependencies with version constraints (see requirements.txt for complete list):

- pandas (>=2.0.0, <2.3.0) - Data manipulation
- numpy (>=1.24.0, <2.0.0) - Numerical operations
- scikit-learn (>=1.3.0, <1.6.0) - Machine learning algorithms
- tensorflow (>=2.16.0, <2.17.0) - Deep learning (optional)
- xgboost (>=2.0.0, <3.0.0) - Gradient boosting
- lightgbm (>=4.0.0, <5.0.0) - Gradient boosting
- imbalanced-learn (>=0.11.0, <0.13.0) - SMOTE implementation
- shap (>=0.44.0, <0.46.0) - Model explanations
- lime (>=0.2.0.1) - Local explanations
- streamlit (>=1.25.0, <2.0.0) - Web dashboard
- fastapi (>=0.100.0, <1.0.0) - REST API

Version constraints ensure reproducibility and compatibility.

## Software Versions

For exact reproduction, document your environment:
- Python version: 3.10+
- Operating system: Windows/macOS/Linux
- Package versions from: pip freeze > environment.txt

## Troubleshooting

### NumPy Version Conflict

Problem: TensorFlow incompatible with NumPy 2.x

Solution:
```bash
python fix_dependencies.py
```

Or manually:
```bash
pip uninstall numpy -y
pip install "numpy>=1.24.0,<2.0.0"
```

### Model Not Found

Problem: Models not trained yet

Solution:
```bash
python src/models/train_cv.py
```

### Dataset Missing

Problem: Raw dataset not downloaded

Solution:
```bash
python download_dataset.py
```

Or download manually from UCI/Kaggle and place at data/raw/maternal_health.csv

## Contributing

This is an academic research project. If you wish to build upon this work:

1. Cite the original work appropriately
2. Maintain the validation methodology standards
3. Document any modifications clearly
4. Consider ethical implications of changes
5. Test thoroughly before any deployment

## License

See LICENSE file for details.

## Acknowledgments

- Dataset from UCI Machine Learning Repository
- Original study by Ahmed et al.
- Built using open-source scientific Python libraries

## Contact and Support

For questions about methodology or reproduction:
1. Review run_instructions.txt
2. Check code documentation
3. Review validation methodology section above

## Disclaimer

This system is for research and educational purposes only. It should NEVER be used as the sole basis for medical decisions. The system:

- Is not a medical device
- Has not undergone clinical validation
- Requires prospective validation studies
- May contain biases from training data
- Should not replace clinical judgment
- Requires qualified medical oversight

All medical decisions must be made by qualified healthcare professionals based on comprehensive clinical assessment.

## Version History

- Version 1.0 (2026-01-12): Initial release with robust cross-validation methodology

## References

1. Ahmed, M. et al. - Original dataset publication
2. UCI Machine Learning Repository - Dataset source
3. SMOTE: Chawla, N.V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique
4. SHAP: Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions
5. LIME: Ribeiro, M.T., et al. (2016). "Why Should I Trust You?": Explaining predictions

For complete citations, see dissertation bibliography.
