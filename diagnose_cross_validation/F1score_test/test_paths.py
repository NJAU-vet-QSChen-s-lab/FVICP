#!/usr/bin/env python3
"""
Simple script to test all file paths before running main analysis
"""

import os
import pandas as pd
import argparse

def test_file_paths(args):
    """Test all required file paths"""
    print("=" * 60)
    print("TESTING FILE PATHS")
    print("=" * 60)
    
    # Test data files
    test_files = {
        'test_b1': args.test_b1 if args.test_b1 else None,
        'test_b2': args.test_b2 if args.test_b2 else None,
        'train_b1': args.train_b1,
        'train_b2': args.train_b2,
    }
    
    print("\n1. Testing data files:")
    available_files = {}
    for name, path in test_files.items():
        if os.path.exists(path):
            try:
                # Try to read a few lines to verify it's readable
                df = pd.read_csv(path, nrows=5)
                print(f"✓ {name}: {path} ({df.shape[0]} rows sample, {df.shape[1]} cols)")
                available_files[name] = path
            except Exception as e:
                print(f"⚠ {name}: File exists but error reading: {e}")
        else:
            print(f"✗ {name}: {path} - NOT FOUND")
    
    # Test model files
    model_files = {
        'b1_model': args.b1_model,
        'b2_model': args.b2_model,
    }
    
    print("\n2. Testing model files:")
    for name, path in model_files.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✓ {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {name}: {path} - NOT FOUND")
    
    # Test original cross-validation logic
    print("\n3. Testing original cross-validation paths:")
    original_paths = {
        'b1_cross_data': args.train_b2,
        'b2_cross_data': args.train_b1,
    }
    
    for name, path in original_paths.items():
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} - NOT FOUND")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    
    # Determine the best cross-validation strategy
    if 'train_b1' in available_files and 'train_b2' in available_files:
        print("✓ Use train files for cross-validation:")
        print("  - b1 model tests on train_b2/fascia_b2_all_train.csv")
        print("  - b2 model tests on train_b1/fascia_b1_all_train.csv")
        return 'train'
    elif 'test_b1' in available_files and 'test_b2' in available_files:
        print("✓ Use test files for cross-validation:")
        print("  - b1 model tests on test_b2/fascia_b2_all_test.csv")
        print("  - b2 model tests on test_b1/fascia_b1_all_test.csv")
        return 'test'
    else:
        print("✗ Insufficient data files for cross-validation")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test existence/readability of required paths")
    parser.add_argument("--train-b1", dest="train_b1", required=True, help="Path to fascia_b1_all_train.csv")
    parser.add_argument("--train-b2", dest="train_b2", required=True, help="Path to fascia_b2_all_train.csv")
    parser.add_argument("--test-b1", dest="test_b1", help="Optional path to fascia_b1_all_test.csv")
    parser.add_argument("--test-b2", dest="test_b2", help="Optional path to fascia_b2_all_test.csv")
    parser.add_argument("--b1-model", dest="b1_model", required=True, help="Path to b1 model .pth")
    parser.add_argument("--b2-model", dest="b2_model", required=True, help="Path to b2 model .pth")
    args = parser.parse_args()

    result = test_file_paths(args)
    print(f"\nRecommended data type: {result}")
