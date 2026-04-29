#!/usr/bin/env python
"""Validate all use cases - for CI/CD pipeline"""

import subprocess
import sys
from pathlib import Path

def validate_all():
    """Run validation on all use cases"""
    script_dir = Path(__file__).parent
    install_script = script_dir / 'install.py'
    
    if not install_script.exists():
        print("❌ install.py not found")
        return False
    
    # Run install.py --all --check
    result = subprocess.run(
        [sys.executable, str(install_script), '--all', '--check'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print("\n❌ Validation FAILED")
        return False
    
    print("\n✅ All use cases validated successfully")
    return True

if __name__ == '__main__':
    success = validate_all()
    sys.exit(0 if success else 1)