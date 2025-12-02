"""
🔍 IPL Streamlit App - Diagnostic & Verification Script
Helps identify and fix deployment issues

Run this script to verify your setup is correct:
    python verify_setup.py
"""

import os
import sys
import pickle
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("\n📌 Checking Python Version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ ERROR: Python 3.8+ required!")
        return False
    else:
        print("   ✅ Python version OK")
        return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📌 Checking Dependencies...")
    required = {
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'joblib': 'Joblib'
    }
    
    all_ok = True
    for package, name in required.items():
        try:
            __import__(package)
            print(f"   ✅ {name} installed")
        except ImportError:
            print(f"   ❌ {name} NOT installed - Run: pip install {package}")
            all_ok = False
    
    return all_ok

def check_data_files():
    """Check if CSV data files exist"""
    print("\n📌 Checking Data Files...")
    required_files = [
        'ipl_matches_data_cleaned.csv',
        'teams_data_cleaned.csv',
        'players_data_cleaned.csv',
        'cleaned_ball_by_ball_data.csv'
    ]
    
    all_ok = True
    for filename in required_files:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"   ✅ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ {filename} NOT FOUND")
            all_ok = False
    
    return all_ok

def check_model_files():
    """Check if trained model files exist"""
    print("\n📌 Checking Model Files...")
    models_dir = 'trained_models'
    required_files = [
        'ipl_model.pkl',
        'ipl_encoders.pkl',
        'model_metadata.pkl'
    ]
    
    if not os.path.exists(models_dir):
        print(f"   ❌ Directory '{models_dir}' does NOT exist!")
        print(f"      Create it and run: python train_and_save_model.py")
        return False
    
    print(f"   ✅ Directory '{models_dir}' exists")
    
    all_ok = True
    for filename in required_files:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   ✅ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ {filename} NOT FOUND")
            all_ok = False
    
    return all_ok

def test_model_loading():
    """Test if we can actually load the model"""
    print("\n📌 Testing Model Loading...")
    try:
        with open(os.path.join('trained_models', 'ipl_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        print("   ✅ Model loaded successfully")
        
        with open(os.path.join('trained_models', 'ipl_encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
        print("   ✅ Encoders loaded successfully")
        
        with open(os.path.join('trained_models', 'model_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        print("   ✅ Metadata loaded successfully")
        print(f"      Model Accuracy: {metadata.get('accuracy', 'N/A'):.2%}")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed to load models: {e}")
        return False

def check_git_status():
    """Check if files are committed to Git"""
    print("\n📌 Checking Git Status...")
    
    if not os.path.exists('.git'):
        print("   ⚠️  Not a Git repository")
        return False
    
    try:
        import subprocess
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        
        uncommitted = result.stdout.strip()
        if uncommitted:
            print("   ⚠️  Uncommitted changes detected:")
            for line in uncommitted.split('\n')[:10]:
                print(f"      {line}")
            return False
        else:
            print("   ✅ All changes committed")
            
            # Check if model files are tracked
            result = subprocess.run(['git', 'ls-files', 'trained_models/'], 
                                  capture_output=True, text=True)
            tracked_files = result.stdout.strip().split('\n')
            
            if any('.pkl' in f for f in tracked_files):
                print("   ✅ Model files tracked in Git")
                return True
            else:
                print("   ❌ Model files NOT tracked in Git")
                print("      Run: git add trained_models/*.pkl && git push")
                return False
    
    except Exception as e:
        print(f"   ⚠️  Could not check Git: {e}")
        return False

def check_streamlit_config():
    """Check if Streamlit config exists"""
    print("\n📌 Checking Streamlit Configuration...")
    
    if os.path.exists('.streamlit/config.toml'):
        print("   ✅ .streamlit/config.toml exists")
        return True
    else:
        print("   ⚠️  .streamlit/config.toml not found (optional)")
        return True

def check_requirements_txt():
    """Check if requirements.txt exists"""
    print("\n📌 Checking Requirements File...")
    
    if os.path.exists('requirements.txt'):
        print("   ✅ requirements.txt exists")
        with open('requirements.txt', 'r') as f:
            packages = f.read().strip().split('\n')
        print(f"      Contains {len(packages)} packages:")
        for pkg in packages[:5]:
            print(f"      - {pkg}")
        return True
    else:
        print("   ❌ requirements.txt NOT found")
        print("      This is required for Streamlit Cloud deployment")
        return False

def generate_report():
    """Run all checks and generate report"""
    print("=" * 60)
    print("🔍 IPL STREAMLIT APP - VERIFICATION REPORT")
    print("=" * 60)
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Data Files': check_data_files(),
        'Model Files': check_model_files(),
        'Model Loading': test_model_loading(),
        'Streamlit Config': check_streamlit_config(),
        'Requirements File': check_requirements_txt(),
        'Git Status': check_git_status(),
    }
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {check}")
    
    print("\n" + "=" * 60)
    
    if passed == total:
        print("✨ ALL CHECKS PASSED! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Test locally: streamlit run ipl_streamlit_app_with_saved_model.py")
        print("  2. Deploy: git push origin main")
        print("  3. On share.streamlit.io, create new app pointing to your repo")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nFor help, see: COMPLETE_SETUP_GUIDE.md")
    
    print("=" * 60 + "\n")

if __name__ == '__main__':
    generate_report()
