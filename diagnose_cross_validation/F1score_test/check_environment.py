#!/usr/bin/env python3
"""
Environment and file checker for F1-score analysis
Run this first to ensure all dependencies and files are available
"""

import os
import sys
import argparse

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = [
        'torch', 'sklearn', 'pandas', 'numpy', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✓ All required packages are installed")
    return True

def check_required_files(paths):
    """Check if all required model and data files exist"""
    required_files = [
        paths.b1_model,
        paths.b2_model,
        paths.b1_data,
        paths.b2_data,
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {os.path.basename(file_path)}")
        else:
            missing_files.append(file_path)
            print(f"✗ {os.path.basename(file_path)} - MISSING")
    
    if missing_files:
        print(f"\nMissing files: {len(missing_files)}")
        for f in missing_files:
            print(f"  - {f}")
        return False
    
    print("✓ All required files are present")
    return True

def check_output_directory(output_dir):
    """Check if output directory exists and is writable"""
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"✓ Created output directory: {output_dir}")
        except Exception as e:
            print(f"✗ Cannot create output directory: {e}")
            return False
    else:
        print(f"✓ Output directory exists: {output_dir}")
    
    # Test write permissions
    test_file = os.path.join(output_dir, 'test_write.tmp')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✓ Output directory is writable")
        return True
    except Exception as e:
        print(f"✗ Cannot write to output directory: {e}")
        return False

def main():
    """Main environment check"""
    parser = argparse.ArgumentParser(description="Check environment and required files")
    parser.add_argument("--b1-data", required=True)
    parser.add_argument("--b2-data", required=True)
    parser.add_argument("--b1-model", required=True)
    parser.add_argument("--b2-model", required=True)
    parser.add_argument("--output-dir", default=os.getcwd())
    args = parser.parse_args()
    print("=" * 50)
    print("F1-SCORE ANALYSIS ENVIRONMENT CHECK")
    print("=" * 50)
    
    print("\n1. Checking Python packages...")
    packages_ok = check_python_packages()
    
    print("\n2. Checking required files...")
    files_ok = check_required_files(args)
    
    print("\n3. Checking output directory...")
    output_ok = check_output_directory(args.output_dir)
    
    print("\n" + "=" * 50)
    if packages_ok and files_ok and output_ok:
        print("✓ ENVIRONMENT CHECK PASSED")
        print("You can now run: python run_analysis.py --help")
    else:
        print("✗ ENVIRONMENT CHECK FAILED")
        print("Please fix the issues above before running the analysis")
    print("=" * 50)
    
    return packages_ok and files_ok and output_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
