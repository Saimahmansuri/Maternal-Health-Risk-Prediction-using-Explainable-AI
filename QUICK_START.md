# Quick Start Guide for Dissertation Results

This guide gets you from zero to dissertation-ready results in 4 steps.

## Prerequisites

- Python 3.10 or higher installed
- Virtual environment activated (recommended)

## Step 1: Install Dependencies (5 minutes)

```bash
pip install -r requirements.txt
```

If you get NumPy version errors:
```bash
python fix_dependencies.py
```

## Step 2: Verify Setup (1 minute)

```bash
python verify_setup.py
```

This checks:
- All packages installed
- All files present
- Random seeds configured
- Directories created

Fix any issues before proceeding.

## Step 3: Process Data (2 minutes)

```bash
python src/data_processing.py
```

This will:
- Download dataset (if needed)
- Clean and process data
- Create train/val/test splits
- Save processed files

## Step 4: Run Cross-Validation Training (5-15 minutes)

```bash
python src/models/train_cv.py
```

This is the main training with proper validation. It will:
- Run stratified 5-fold cross-validation
- Compare SMOTE vs no resampling
- Report mean ± std for all metrics
- Select best model based on High-Risk recall
- Train final model on full dataset
- Save everything for dissertation

## Results Location

After Step 4, check these files:

**For Dissertation Tables:**
- `reports/cv_results_summary.csv` - Main results table
- `reports/cv_results_folds.csv` - Fold-by-fold details

**For Model Info:**
- `models/model_metadata.json` - Best model configuration
- `models/best_model.pkl` - Trained model

**For Comparison:**
- `reports/resampling_strategy_comparison.csv` - SMOTE analysis

## Optional: Launch Dashboard (1 minute)

To test the system interactively:

```bash
streamlit run dashboard/streamlit_app.py
```

Access at: http://localhost:8501

## What to Report in Dissertation

### Methods Section

```
Data Validation:
- Stratified 5-fold cross-validation
- SMOTE applied only within training folds (no data leakage)
- Random seed: 42 for reproducibility
- Models evaluated: LightGBM, XGBoost, Gradient Boosting
- Selection criterion: High-Risk recall (clinical priority)
```

### Results Section

From `cv_results_summary.csv`:

```
Best Model: [Model Name]
Strategy: [SMOTE or No Resampling]

Performance (mean ± std, 5-fold CV):
- Accuracy: X.XXXX ± X.XXXX
- F1 Score (weighted): X.XXXX ± X.XXXX
- F1 Score (macro): X.XXXX ± X.XXXX
- High-Risk Recall: X.XXXX ± X.XXXX
```

From `models/model_metadata.json`:

```
Final Model Test Set Performance:
- Test Accuracy: X.XXXX
- Test F1 (weighted): X.XXXX
- Test High-Risk Recall: X.XXXX
```

### Limitations Section

Copy from `README.md` Limitations section:
- Small dataset size (n~1000)
- Single geographic region
- No longitudinal data
- Requires prospective validation
- Potential demographic bias
- etc.

## Troubleshooting

**Problem:** Dataset not found
**Solution:** 
```bash
python download_dataset.py
```
Or download manually from Kaggle/UCI and place at `data/raw/maternal_health.csv`

**Problem:** Import errors
**Solution:**
```bash
pip install -r requirements.txt --force-reinstall
```

**Problem:** NumPy version conflict
**Solution:**
```bash
python fix_dependencies.py
```

**Problem:** Different results each time
**Check:**
- Random seed is 42 in all files
- Using same package versions
- Not using GPU (use CPU for exact reproduction)

## Time Estimate

- First-time setup: 15-30 minutes
- Data processing: 2 minutes
- CV training: 5-15 minutes
- Total: ~20-45 minutes

## What's Different from Before

**Old approach (train.py):**
- Single train/val/test split
- Risk of lucky split
- Optimistic results
- No variability reporting

**New approach (train_cv.py):**
- 5-fold cross-validation
- Robust to data splits
- Mean ± std reporting
- No data leakage
- Scientifically defensible

## Important Files

**Must read:**
1. `CHANGES_SUMMARY.md` - What changed and why
2. `run_instructions.txt` - Detailed reproduction guide
3. `README.md` - Complete documentation

**Must run:**
1. `src/models/train_cv.py` - The main training script

**Must check:**
1. `reports/cv_results_summary.csv` - Your results
2. `models/model_metadata.json` - Model info

## Final Checklist Before Submission

- [ ] Ran `python src/models/train_cv.py` successfully
- [ ] Generated `cv_results_summary.csv` and `cv_results_folds.csv`
- [ ] Copied results into dissertation tables
- [ ] Updated methods section with validation approach
- [ ] Added limitations from README
- [ ] Made claims conservative (pilot validation, not deployment)
- [ ] Cited random seed and reproducibility approach
- [ ] Acknowledged dataset size and geographic limitations

## Questions?

1. **How do I know which model is best?**
   - Check `models/model_metadata.json` - it's listed there
   - Also printed at end of train_cv.py output

2. **Should I report CV or test set results?**
   - Report both
   - CV shows robustness across splits
   - Test set shows final performance on held-out data

3. **What if my results differ slightly from expected?**
   - Small variations are normal due to system differences
   - Overall conclusions should be stable
   - Document your exact environment

4. **Do I need to run the old train.py?**
   - No, train_cv.py is sufficient
   - train.py kept for backward compatibility only

5. **How long does training take?**
   - Depends on hardware
   - Typically 5-15 minutes for CV training
   - Be patient, it's doing 5 folds × 3 models × 2 strategies = 30 training runs

## Support

If something doesn't work:
1. Run `python verify_setup.py` first
2. Check `run_instructions.txt` for detailed steps
3. Review error messages carefully
4. Ensure all dependencies installed correctly



