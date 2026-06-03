#!/usr/bin/env python
"""
Extract Python versions from all use case config files for CI matrix.
Outputs JSON format for GitHub Actions.
"""

import json
from pathlib import Path
import yaml


def main():
    script_dir = Path(__file__).parent
    use_cases = []
    
    for item in sorted(script_dir.iterdir()):
        if item.is_dir() and (item / 'config.yaml').exists():
            config_path = item / 'config.yaml'
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            python_version = config.get('python_version', '3.13')
            use_case = config['use_case']
            
            use_cases.append({
                'use_case': use_case,
                'python_version': python_version
            })
    
    print(json.dumps(use_cases))


if __name__ == '__main__':
    main()