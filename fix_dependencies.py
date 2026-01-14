"""
Comprehensive Dependency Fixer for Maternal Health Risk Prediction System
Fixes version conflicts across main project and Timeline modules
"""

import subprocess
import sys
import os
from pathlib import Path

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    import pkg_resources
except ImportError:
    print("Installing setuptools...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])
    import pkg_resources

def get_installed_version(package_name):
    """Get the currently installed version of a package"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def check_version_compatible(installed_version, required_spec):
    """Check if installed version meets requirements"""
    try:
        pkg_resources.require(f"{required_spec}")
        return True
    except (pkg_resources.VersionConflict, pkg_resources.DistributionNotFound):
        return False

def fix_dependencies():
    """Fix all dependency conflicts in the project"""
    
    print("="*80)
    print("COMPREHENSIVE DEPENDENCY FIXER")
    print("Maternal Health Risk Prediction System")
    print("="*80)
    
    # Critical packages that need specific versions
    critical_packages = [
        ("numpy", "numpy>=1.24.0,<2.0.0", "NumPy 2.x has breaking changes"),
        ("pandas", "pandas>=2.0.0,<2.3.0", "Updated pandas compatibility"),
        ("scikit-learn", "scikit-learn>=1.3.0,<1.6.0", "Ensure sklearn compatibility"),
        ("tensorflow", "tensorflow>=2.16.0,<2.17.0", "TensorFlow with NumPy 1.x support"),
        ("keras", "keras>=3.0.0,<4.0.0", "Keras 3.x compatibility"),
        ("ml-dtypes", "ml-dtypes>=0.3.1,<0.5.0", "TensorFlow dtypes compatibility"),
        ("shap", "shap>=0.44.0,<0.46.0", "Explainable AI - SHAP"),
        ("lime", "lime>=0.2.0.1", "Explainable AI - LIME"),
        ("xgboost", "xgboost>=2.0.0,<3.0.0", "XGBoost compatibility"),
        ("lightgbm", "lightgbm>=4.0.0,<5.0.0", "LightGBM compatibility"),
    ]
    
    print("\n" + "="*80)
    print("PHASE 1: Checking Critical Package Versions")
    print("="*80)
    
    packages_to_fix = []
    
    for package_name, package_spec, reason in critical_packages:
        print(f"\n[*] Checking {package_name}...")
        installed = get_installed_version(package_name)
        
        if installed is None:
            print(f"    [X] Not installed")
            print(f"    [!] Reason: {reason}")
            packages_to_fix.append(package_spec)
        else:
            print(f"    [i] Installed version: {installed}")
            if not check_version_compatible(installed, package_spec):
                print(f"    [!] Version conflict detected!")
                print(f"    [!] Reason: {reason}")
                packages_to_fix.append(package_spec)
            else:
                print(f"    [OK] Version compatible")
    
    if not packages_to_fix:
        print("\n" + "="*80)
        print("[SUCCESS] ALL DEPENDENCIES ARE COMPATIBLE!")
        print("="*80)
        return
    
    print("\n" + "="*80)
    print("PHASE 2: Fixing Package Conflicts")
    print("="*80)
    print(f"\nFound {len(packages_to_fix)} package(s) that need to be fixed.\n")
    
    print("[WARNING] This will modify your Python environment!")
    print("It's recommended to use a virtual environment.\n")
    
    response = input("Do you want to proceed with the fixes? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n[CANCELLED] Dependency fix cancelled by user.")
        print("\nTo fix manually, run:")
        print("  pip install -r requirements.txt --upgrade")
        return
    
    print("\n" + "-"*80)
    print("Installing/Upgrading packages...")
    print("-"*80)
    
    # Install all fixes at once for better dependency resolution
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + packages_to_fix + ["--upgrade"]
        print(f"\nRunning: {' '.join(cmd)}\n")
        subprocess.check_call(cmd)
        
        print("\n" + "="*80)
        print("[SUCCESS] DEPENDENCIES FIXED SUCCESSFULLY!")
        print("="*80)
        
        # Verify fixes
        print("\nVerifying fixes...")
        all_ok = True
        for package_name, package_spec, _ in critical_packages:
            installed = get_installed_version(package_name)
            if installed and check_version_compatible(installed, package_spec):
                print(f"  [OK] {package_name} {installed}")
            else:
                print(f"  [X] {package_name} still has issues")
                all_ok = False
        
        if all_ok:
            print("\n[SUCCESS] All critical packages are now compatible!")
        else:
            print("\n[WARNING] Some packages may still have issues.")
            print("Try running: pip install -r requirements.txt --force-reinstall")
            
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Error during installation: {e}")
        print("\nTry fixing manually:")
        print("  1. Create a fresh virtual environment")
        print("  2. Run: pip install -r requirements.txt")
        return
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Main Project:")
    print("   python run_pipeline.py")
    print("\n2. Timeline Module 1 (Data & ML Core):")
    print("   cd Timeline/project_modules/Module_1_Data_ML_Core")
    print("   pip install -r requirements.txt")
    print("   python run_pipeline.py")
    print("\n3. Timeline Module 2 (Prediction Service):")
    print("   cd Timeline/project_modules/Module_2_Prediction_Service")
    print("   pip install -r requirements.txt")
    print("\n4. Timeline Module 3 (User Interfaces):")
    print("   cd Timeline/project_modules/Module_3_User_Interfaces")
    print("   pip install -r requirements.txt")
    print("   python run_dashboard.py")
    print("\n" + "="*80)

def check_all_modules():
    """Check requirements for all modules"""
    print("\n" + "="*80)
    print("SCANNING ALL MODULES")
    print("="*80)
    
    requirements_files = [
        ("Main Project", Path("requirements.txt")),
        ("Module 1: Data & ML Core", Path("Timeline/project_modules/Module_1_Data_ML_Core/requirements.txt")),
        ("Module 2: Prediction Service", Path("Timeline/project_modules/Module_2_Prediction_Service/requirements.txt")),
        ("Module 3: User Interfaces", Path("Timeline/project_modules/Module_3_User_Interfaces/requirements.txt")),
    ]
    
    for module_name, req_file in requirements_files:
        if req_file.exists():
            print(f"\n  [OK] {module_name}: {req_file}")
        else:
            print(f"\n  [X] {module_name}: {req_file} - NOT FOUND")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print("\nMATERNAL HEALTH RISK PREDICTION - DEPENDENCY FIXER")
    print("This tool will fix version conflicts across all modules.\n")
    
    # Check if running in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("[WARNING]  WARNING: You're not in a virtual environment!")
        print("It's strongly recommended to use a virtual environment.\n")
    
    check_all_modules()
    fix_dependencies()
