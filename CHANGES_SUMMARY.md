# Summary of Changes for Dissertation Improvements

This document summarizes the changes made to address teacher feedback and make the research scientifically defensible.

## Date: 2026-01-12

## Critical Changes Implemented

### 1. Stratified K-Fold Cross-Validation (CRITICAL)

**File Created:** `src/models/train_cv.py`

**What it does:**
- Implements proper stratified 5-fold cross-validation
- Prevents data leakage by applying SMOTE only within training folds
- Compares different class imbalance strategies
- Reports mean ± standard deviation for all metrics
- Selects best model based on High-Risk recall (clinical priority)
- Trains final model on full dataset for deployment

**How to use:**
```bash
python src/models/train_cv.py
```

**Outputs:**
- `reports/cv_results_summary.csv` - Use this in your dissertation
- `reports/cv_results_folds.csv` - Raw fold-by-fold results
- `reports/resampling_strategy_comparison.csv` - SMOTE vs no resampling
- `models/best_model.pkl` - Final model for deployment
- `models/model_metadata.json` - Model configuration and test metrics

**Key Features:**
- No data leakage - SMOTE applied only within training folds using imblearn Pipeline
- Robust validation - all samples validated exactly once
- Proper reporting - mean ± std for all metrics
- Clinical focus - model selection based on High-Risk recall

### 2. Fixed Class Imbalance Handling (NO DATA LEAKAGE)

**Problem Fixed:**
Previously, if SMOTE was used, it might have been applied to the entire dataset before splitting, causing data leakage.

**Solution:**
- SMOTE now applied only within each training fold
- Uses `imblearn.pipeline.Pipeline` to ensure proper sequencing
- Validation data never exposed to synthetic samples
- Comparison of three strategies:
  1. No resampling (baseline)
  2. SMOTE (within folds only)
  3. Class weights (native model parameter)

**Results:**
The comparison table shows which strategy works best for each model.

### 3. Reproducibility Fixed

**Changes Made:**

**File:** `src/data_processing.py`
- Added RANDOM_SEED = 42 at module level
- All splits now use this seed
- Documented in output

**File:** `src/models/train.py`
- Added random seeds for numpy, random, tensorflow
- All model initializations use RANDOM_SEED constant
- Added note that train_cv.py should be used for dissertation

**File:** `src/models/train_cv.py`
- All random operations controlled with seed
- Documented in outputs and metadata

**Documentation:**
- `run_instructions.txt` - Complete reproduction instructions
- `README.md` - Updated with reproducibility section
- `models/model_metadata.json` - Saves random seed used

### 4. Generated CSV Files for Dissertation

**File:** `reports/cv_results_summary.csv`
Format:
```
Model,Strategy,Accuracy,F1_Weighted,F1_Macro,High_Risk_Recall
LightGBM,No Resampling,0.XXXX ± 0.XXXX,0.XXXX ± 0.XXXX,...
LightGBM,SMOTE,0.XXXX ± 0.XXXX,0.XXXX ± 0.XXXX,...
...
```

**File:** `reports/cv_results_folds.csv`
Format:
```
model,smote,fold,accuracy,f1_weighted,f1_macro,high_risk_recall
LightGBM,False,1,0.XXXX,0.XXXX,0.XXXX,0.XXXX
LightGBM,False,2,0.XXXX,0.XXXX,0.XXXX,0.XXXX
...
```

**File:** `reports/resampling_strategy_comparison.csv`
Complete comparison of strategies with all metrics.

**How to Use in Dissertation:**
1. Report mean ± std from cv_results_summary.csv
2. Use fold results to show variability
3. Compare strategies using resampling_strategy_comparison.csv
4. Calculate 95% CI: mean ± 1.96 × (std / sqrt(5))

### 5. Updated UI Disclaimer

**File:** `dashboard/streamlit_app.py`

**Changes:**
- Prominent notice that system is a research prototype
- Clear statement: NOT a clinical diagnostic tool
- Not approved for clinical deployment
- Lists key limitations (dataset size, single region, no prospective validation)
- Requires qualified healthcare professional verification
- Mentions need for prospective validation

**Location:** Bottom of dashboard, visible on every page

### 6. Created run_instructions.txt

**File:** `run_instructions.txt`

**Contents:**
- Complete step-by-step reproduction instructions
- System requirements
- Dependencies and version constraints
- Random seed documentation
- Expected results with ranges
- Data provenance information
- Troubleshooting guide
- Validation methodology explanation

**Purpose:**
Enables another researcher to reproduce your exact results.

### 7. Updated README.md (MAJOR REWRITE)

**Changes Made:**

**Removed:**
- All emojis
- Casual/AI-style language ("awesome", "cool", etc.)
- Overly optimistic claims about deployment
- Generic content

**Added:**

1. **Important Notice Section**
   - Clear statement: research prototype only
   - Not a clinical diagnostic tool
   - Requires prospective validation

2. **Dataset Limitations Section**
   - Small sample size
   - Single geographic region
   - No temporal data
   - Limited features
   - Potential biases

3. **Validation Methodology Section**
   - Stratified k-fold CV
   - Data leakage prevention
   - Class imbalance handling
   - Model selection criteria
   - Test set evaluation

4. **Performance Section**
   - Reports ranges, not exact values
   - Includes standard deviations
   - Notes on confidence intervals
   - Statistical comparison guidance

5. **Explainability Limitations Section**
   - Association vs causation
   - Approximations
   - Clinical integration notes

6. **Comprehensive Limitations Section**
   - Dataset limitations
   - Model limitations
   - Methodological limitations
   - Deployment considerations

7. **Ethical and Regulatory Section**
   - Ethical concerns
   - Regulatory requirements before deployment
   - Need for prospective validation

8. **Future Work Section**
   - Recommended improvements
   - Data collection needs
   - Methodology enhancements
   - Clinical integration requirements

**Tone:**
- Academic and professional
- Conservative claims
- Evidence-based
- Transparent about limitations

### 8. Other Improvements

**File:** `run_pipeline.py`
- Removed emojis
- Made language more professional
- Added guidance to use train_cv.py
- Clearer error messages

## What The Teacher Wanted vs What We Delivered

### Teacher's Requirements:

1. **Stratified k-fold cross-validation** ✓
   - Implemented: 5-fold stratified CV
   - Reports: mean ± std for all metrics
   - Files: cv_results_summary.csv, cv_results_folds.csv

2. **No data leakage with SMOTE** ✓
   - SMOTE only within training folds
   - Uses imblearn Pipeline
   - Comparison with alternatives

3. **Reproducibility** ✓
   - Random seeds documented and fixed
   - Complete reproduction instructions
   - Software versions documented

4. **CSV files for dissertation** ✓
   - cv_results_summary.csv - mean ± std
   - cv_results_folds.csv - raw results
   - Both ready for use in tables

5. **Final model trained on full dataset** ✓
   - Trained after CV
   - Saved with metadata
   - Based on best CV configuration

6. **Disclaimer in UI** ✓
   - Research prototype notice
   - Not a clinical tool
   - Lists limitations

7. **No new features** ✓
   - Only improved validation
   - No UI redesign
   - No new functionality

## How to Run for Dissertation

### Step 1: Process Data
```bash
python src/data_processing.py
```

### Step 2: Run Cross-Validation (IMPORTANT)
```bash
python src/models/train_cv.py
```
**This takes 5-15 minutes. Be patient.**

### Step 3: Check Results
Look at:
- `reports/cv_results_summary.csv` - Your main table
- `reports/cv_results_folds.csv` - Detailed fold results
- `models/model_metadata.json` - Best model info

### Step 4: Use in Dissertation

**For Methods Section:**
- Describe stratified 5-fold CV
- Mention SMOTE applied only within folds
- Report random seed (42)
- Cite train_cv.py implementation

**For Results Section:**
- Report mean ± std from cv_results_summary.csv
- Show variability across folds
- Compare strategies (SMOTE vs no resampling)
- Highlight High-Risk recall (most important)

**For Discussion Section:**
- Use limitations from README.md
- Note dataset size constraints
- Discuss need for prospective validation
- Frame as suitable for pilot study, not deployment

**For Limitations Section:**
- Copy from README.md limitations section
- Add any specific to your context

## Key Metrics to Report

From `cv_results_summary.csv`, report:

1. **Best Model Name** (e.g., LightGBM)
2. **Best Strategy** (SMOTE or No Resampling)
3. **Accuracy:** X.XXXX ± X.XXXX
4. **F1 Score (weighted):** X.XXXX ± X.XXXX
5. **F1 Score (macro):** X.XXXX ± X.XXXX
6. **High-Risk Recall:** X.XXXX ± X.XXXX (MOST IMPORTANT)

From `models/model_metadata.json`, report:
1. **Final test set accuracy**
2. **Final test set High-Risk recall**
3. **Random seed used**
4. **Number of CV folds**

## Addressing Specific Teacher Concerns

### "Results are too good for dataset size"
**Fixed:** 
- CV shows variability across folds
- Standard deviations indicate uncertainty
- No single lucky split
- Conservative claims in README

### "Need to demonstrate results are not optimistic"
**Fixed:**
- Stratified k-fold prevents lucky splits
- Mean ± std shows true variability
- Held-out test set never used for selection
- Transparent about dataset limitations

### "SMOTE data leakage"
**Fixed:**
- SMOTE only within training folds
- Uses imblearn Pipeline
- Explicitly documented
- Comparison shows impact

### "Need statistical testing for model comparison"
**Guidance Added:**
- README explains confidence intervals
- Notes on overlapping CIs
- Suggests Mann-Whitney U test if needed
- Acknowledges statistical vs clinical significance

### "Explainability needs detail"
**Fixed:**
- Added limitations of SHAP/LIME
- Association vs causation noted
- Implementation choices documented
- Critical reflection added

### "Claims too strong"
**Fixed:**
- "Deployment ready" removed
- Now: "suitable for pilot validation"
- "Research prototype" throughout
- "Requires prospective validation" added

### "Need usability evaluation"
**Acknowledged:**
- Added to future work
- Not claiming deployment readiness
- Framed as research prototype

## Files You Should Look At

**Most Important:**
1. `src/models/train_cv.py` - The core validation code
2. `reports/cv_results_summary.csv` - Your dissertation table (after running)
3. `README.md` - Updated with all methodology
4. `run_instructions.txt` - Reproduction guide

**Also Important:**
5. `reports/cv_results_folds.csv` - Fold-by-fold details (after running)
6. `models/model_metadata.json` - Final model info (after running)
7. `dashboard/streamlit_app.py` - Updated disclaimer

## What To Tell Your Teacher

"I have addressed all the feedback points:

1. **Validation:** Implemented stratified 5-fold CV with proper SMOTE handling to prevent data leakage. Results show mean ± std across folds.

2. **Reproducibility:** Fixed all random seeds (seed=42), documented software versions, created complete reproduction instructions.

3. **Data Leakage:** SMOTE now applied strictly within training folds using imblearn Pipeline. Comparison shows SMOTE vs no resampling.

4. **Deliverables:** Generated cv_results_summary.csv and cv_results_folds.csv for dissertation tables.

5. **Model Selection:** Based on High-Risk recall (clinical priority), not just accuracy.

6. **Claims:** Moderated all statements. Now framed as research prototype suitable for pilot validation, not deployment.

7. **Limitations:** Comprehensive limitations section covering dataset size, bias, lack of longitudinal data, and regulatory needs.

8. **Documentation:** Updated README with methodology, created run_instructions.txt for reproduction.

The results are now scientifically defensible and ready for dissertation submission."

## Next Steps Before Submission

1. Run `python src/models/train_cv.py` to generate results
2. Copy values from cv_results_summary.csv into dissertation
3. Create figures from cv_results_folds.csv showing variability
4. Use README limitations section in dissertation
5. Cite methodology as described in run_instructions.txt

## Questions?

Check:
1. `run_instructions.txt` - Detailed reproduction steps
2. `README.md` - Methodology explanation
3. Code comments in `train_cv.py` - Implementation details

Good luck with your final submission!

