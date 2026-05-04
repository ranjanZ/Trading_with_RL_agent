#!/usr/bin/env python3
"""Run all tests for the trading RL system"""

import sys
import subprocess
from pathlib import Path

def run_test(module_path: str):
    """Run a single test module"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {module_path}")
    print(f"{'='*80}")
    
    result = subprocess.run(
        [sys.executable, module_path],
        capture_output=False,
        text=True
    )
    return result.returncode

def main():
    """Run all test modules"""
    
    test_modules = [
        "data/connectors/__init__.py",
        "data/connectors/mt5_connector.py",
        "data/data_loader.py",
        "data/feature_engineering.py",
        "environments/trading_env.py",
        "models/policy_network.py",
        "training/imitation_learning.py",
        "training/rl_trainer.py",
        "deployment/live_trader.py",
        "utils/visualization.py",
        "utils/metrics.py"
    ]
    
    passed = 0
    failed = 0
    
    for module in test_modules:
        path = Path(module)
        if path.exists():
            code = run_test(module)
            if code == 0:
                passed += 1
            else:
                failed += 1
        else:
            print(f"\n⚠️ Module not found: {module}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the output above.")

if __name__ == "__main__":
    main()