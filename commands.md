# Command Reference Guide

This is a quick reference for all the commands you'll need to work with this project.

## Initial Setup Commands

### Creating a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

You'll know the virtual environment is active when you see `(venv)` at the start of your command prompt.

### Installing Everything

**Basic installation:**

```bash
pip install -r requirements.txt
```

This installs all the packages the project needs. Takes about 5-10 minutes depending on your internet speed.

**If you get version conflicts (especially NumPy/TensorFlow issues):**

```bash
python fix_dependencies.py
```

This script automatically fixes the common version conflicts that pop up.

**Force reinstall if things are really broken:**

```bash
pip install -r requirements.txt --force-reinstall
```

This removes and reinstalls everything from scratch. Use this as a last resort.

**Install as a package (optional, for development):**

```bash
pip install -e .
```

This makes imports cleaner but isn't required.

## Getting the Dataset

**Automatic download (easiest way):**

```bash
python download_dataset.py
```

This downloads the dataset automatically and saves it to `data/raw/maternal_health.csv`.

**Force download even if file exists:**

```bash
python download_dataset.py --force
```

**Alternative script:**

```bash
python src/download_dataset.py
```

If the download fails, both scripts will create synthetic data so you can still test things.

## Running the Complete Pipeline

**Run everything at once:**

```bash
python run_pipeline.py
```

This does it all:

- Downloads dataset if missing
- Processes and cleans the data
- Trains all 7 models
- Generates evaluation reports
- Creates explainability visualizations

Takes 5-10 minutes. Just let it run and grab a coffee.

## Running Individual Components

Sometimes you don't want to run everything. Here's how to run specific parts:

**Just data processing:**

```bash
python src/data_processing.py
```

This loads, cleans, and splits the data without training models.

**Just model training:**

```bash
python src/models/train.py
```

Trains all models. Assumes data is already processed.

**Just make a prediction:**

```bash
python src/models/predict.py
```

Runs an example prediction. Assumes models are already trained.

**Just generate explainability reports:**

```bash
python src/explainers.py
```

Creates SHAP and LIME visualizations. Assumes models are trained.

## Running the Web Dashboard

**Start the Streamlit dashboard:**

```bash
streamlit run dashboard/streamlit_app.py
```

Then open your browser to: http://localhost:8501

**If port 8501 is already being used:**

```bash
streamlit run dashboard/streamlit_app.py --server.port 8502
```

Change 8502 to any available port number.

**Run without watching for file changes (faster):**

```bash
streamlit run dashboard/streamlit_app.py --server.runOnSave false
```

**Stop the dashboard:**
Just press `Ctrl+C` in the terminal.

## Running the API

**Start the FastAPI server:**

```bash
python src/api/app.py
```

Then open your browser to: http://localhost:8000/docs

**Alternative way to run it:**

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Run on a different port:**

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8001 --reload
```

**Run without auto-reload (for production):**

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

**Stop the API:**
Press `Ctrl+C` in the terminal.

## Jupyter Notebook Commands

**Start Jupyter:**

```bash
jupyter notebook
```

Your browser will open automatically.

**Start on a different port:**

```bash
jupyter notebook --port 8889
```

**List running notebooks:**

```bash
jupyter notebook list
```

**Stop Jupyter:**
Press `Ctrl+C` twice in the terminal.

## Testing and Verification Commands

**Check your Python version:**

```bash
python --version
```

You need 3.10 or higher.

**Verify environment setup:**

```bash
python check_env.py
```

**Quick import test:**

```bash
python -c "from src.models.predict import MaternalRiskPredictor; print('Everything works!')"
```

**Check installed packages:**

```bash
pip list
```

**Check specific package version:**

```bash
pip show numpy
pip show tensorflow
pip show scikit-learn
```

## Fixing Common Issues

### NumPy/TensorFlow Version Conflicts

**Automatic fix:**

```bash
python fix_dependencies.py
```

**Manual fix:**

```bash
pip uninstall numpy -y
pip install "numpy>=1.24.0,<2.0.0"
pip install "tensorflow>=2.16.0,<2.17.0"
```

### Missing Packages

**Install missing package:**

```bash
pip install shap lime
```

**Reinstall all dependencies:**

```bash
pip install -r requirements.txt --upgrade
```

### Clearing Cache

**Clear pip cache:**

```bash
pip cache purge
```

**Clear Python cache files:**

Windows:

```bash
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
```

Mac/Linux:

```bash
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```

### Port Issues

**Find what's using a port (Windows):**

```bash
netstat -ano | findstr :8000
```

**Find what's using a port (Mac/Linux):**

```bash
lsof -i :8000
```

**Kill process by ID:**

Windows:

```bash
taskkill /PID <process_id> /F
```

Mac/Linux:

```bash
kill -9 <process_id>
```

## Making API Requests

### Using curl

**Health check:**

```bash
curl http://localhost:8000/health
```

**Make a prediction:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Age": 28, "SystolicBP": 140, "DiastolicBP": 90, "BS": 12.0, "BodyTemp": 98.6, "HeartRate": 88}'
```

**Get feature info:**

```bash
curl http://localhost:8000/features
```

**Get model info:**

```bash
curl http://localhost:8000/model/info
```

### Using Python

**Quick prediction test:**

```bash
python -c "from src.models.predict import predict_from_values; result = predict_from_values(25, 120, 80, 7.5, 98.6, 72); print(result['risk_level'])"
```

## Git Commands (If You're Version Controlling)

**Check what changed:**

```bash
git status
```

**Add files to staging:**

```bash
git add .
```

**Commit changes:**

```bash
git commit -m "Your message here"
```

**Push to repository:**

```bash
git push origin main
```

**Discard local changes:**

```bash
git restore <filename>
```

**Discard all local changes:**

```bash
git restore .
```

## Package Management Commands

**Update pip itself:**

```bash
python -m pip install --upgrade pip
```

**Create requirements file from current environment:**

```bash
pip freeze > requirements_backup.txt
```

**Uninstall all packages (nuclear option):**

```bash
pip freeze > temp.txt
pip uninstall -r temp.txt -y
del temp.txt
```

Then reinstall:

```bash
pip install -r requirements.txt
```

## Performance and Monitoring

**Check memory usage while running:**

Windows:

```bash
tasklist | findstr python
```

Mac/Linux:

```bash
ps aux | grep python
```

**Monitor GPU usage (if using GPU for TensorFlow):**

```bash
nvidia-smi
```

Run this in a separate terminal to watch in real-time:

```bash
watch -n 1 nvidia-smi
```

## Cleanup Commands

**Remove generated model files:**

Windows:

```bash
del /s models\*.pkl
del /s models\*.h5
```

Mac/Linux:

```bash
rm -f models/*.pkl
rm -f models/*.h5
```

**Remove processed data:**

Windows:

```bash
del /s data\processed\*.*
```

Mac/Linux:

```bash
rm -f data/processed/*
```

**Remove all reports:**

Windows:

```bash
del /s reports\*.png
del /s reports\*.csv
del /s reports\*.json
```

Mac/Linux:

```bash
rm -f reports/*.png reports/*.csv reports/*.json
```

**Start completely fresh (keep only source code):**

```bash
# Remove generated files
rm -rf models/* data/processed/* reports/*

# Reinstall packages
pip install -r requirements.txt

# Run pipeline again
python run_pipeline.py
```

## Quick Workflow Commands

### First Time Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Mac/Linux

# 2. Install packages
pip install -r requirements.txt

# 3. Fix any conflicts
python fix_dependencies.py

# 4. Run everything
python run_pipeline.py

# 5. Start dashboard
streamlit run dashboard/streamlit_app.py
```

### Daily Development Workflow

```bash
# Activate environment
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Mac/Linux



# Test changes
python src/models/train.py

# If everything works, commit
git add .
git commit -m "Your changes"
git push
```

### Troubleshooting Workflow

```bash
# 1. Check environment
python check_env.py

# 2. Fix dependencies
python fix_dependencies.py

# 3. Reinstall if needed
pip install -r requirements.txt --force-reinstall

# 4. Clear cache
pip cache purge

# 5. Try again
python run_pipeline.py
```

## Pro Tips

**Run command in background (Mac/Linux):**

```bash
python run_pipeline.py &
```

**Run command and save output to file:**

```bash
python run_pipeline.py > output.log 2>&1
```

**Run command with timestamp:**

```bash
python run_pipeline.py | ts
```

**Run with time measurement:**

```bash
time python run_pipeline.py
```

**Multiple commands in sequence:**

```bash
python src/data_processing.py && python src/models/train.py && python src/explainers.py
```

**Run only if previous succeeds:**

```bash
pip install -r requirements.txt && python run_pipeline.py
```

## Environment Variables (If Needed)

**Set environment variable (Windows):**

```bash
set PYTHONPATH=E:\Maternal
set TF_CPP_MIN_LOG_LEVEL=2
```

**Set environment variable (Mac/Linux):**

```bash
export PYTHONPATH=/path/to/Maternal
export TF_CPP_MIN_LOG_LEVEL=2
```

**Load from .env file:**

```bash
python -c "from dotenv import load_dotenv; load_dotenv()"
```

## Getting Help

**See all streamlit options:**

```bash
streamlit run --help
```

**See all uvicorn options:**

```bash
uvicorn --help
```

**See Python path:**

```bash
python -c "import sys; print('\n'.join(sys.path))"
```

**See installed package location:**

```bash
python -c "import pandas; print(pandas.__file__)"
```

---

## Quick Reference Summary

| What You Want to Do   | Command                                    |
| --------------------- | ------------------------------------------ |
| Install everything    | `pip install -r requirements.txt`          |
| Fix version issues    | `python fix_dependencies.py`               |
| Download dataset      | `python download_dataset.py`               |
| Run complete pipeline | `python run_pipeline.py`                   |
| Start web dashboard   | `streamlit run dashboard/streamlit_app.py` |
| Start API             | `python src/api/app.py`                    |
| Train models only     | `python src/models/train.py`               |
| Process data only     | `python src/data_processing.py`            |
| Check environment     | `python check_env.py`                      |
| Open Jupyter          | `jupyter notebook`                         |

---

**Note:** Always make sure your virtual environment is activated before running these commands! You'll see `(venv)` in your terminal prompt when it's active.

If you're ever stuck, the most reliable sequence is:

1. `python fix_dependencies.py`
2. `python run_pipeline.py`
3. `streamlit run dashboard/streamlit_app.py`

That should get you up and running 99% of the time.
