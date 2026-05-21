#!/usr/bin/env python
"""Validate a single use case in isolation.

This script is called as a subprocess to provide complete module isolation.
"""

import sys
import importlib.util
import inspect
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent


def validate_single(use_case: str):
    """Validate a single use case."""
    use_case_dir = SCRIPT_DIR / use_case
    config_path = use_case_dir / 'config.yaml'
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    sys.path.insert(0, str(use_case_dir))
    
    env_config = config['environment']
    env_file = env_config['filepaths']['environment'].split('/')[-1]
    env_path = use_case_dir / env_file
    
    if not env_path.exists():
        print(f"❌ Missing environment: {env_path}", file=sys.stderr)
        sys.exit(1)
    
    env_module_name = f"{use_case}_environment"
    spec = importlib.util.spec_from_file_location(env_module_name, env_path)
    env_mod = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(env_mod)
    except Exception as e:
        print(f"❌ Failed to load {env_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not hasattr(env_mod, 'environment'):
        print(f"❌ {env_path} must define 'environment' variable", file=sys.stderr)
        sys.exit(1)
    
    env = env_mod.environment
    required_methods = ['reset', 'step', 'render']
    for method in required_methods:
        if not hasattr(env, method):
            print(f"❌ environment must have {method}() method", file=sys.stderr)
            sys.exit(1)
    
    print(f"  ✓ Environment validated: {env_config['name']}")
    
    pol_mod = None
    if 'policy' in config:
        pol_config = config['policy']
        
        for key, filepath in pol_config['filepaths'].items():
            pol_file = filepath.split('/')[-1]
            pol_path = use_case_dir / pol_file
            if not pol_path.exists():
                print(f"❌ Missing {key}: {pol_path}", file=sys.stderr)
                sys.exit(1)
        
        pol_file = pol_config['filepaths']['policy'].split('/')[-1]
        pol_path = use_case_dir / pol_file
        
        pol_module_name = f"{use_case}_policy"
        spec = importlib.util.spec_from_file_location(pol_module_name, pol_path)
        pol_mod = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(pol_mod)
        except Exception as e:
            print(f"❌ Failed to load {pol_path}: {e}", file=sys.stderr)
            sys.exit(1)
        
        if not hasattr(pol_mod, 'policy'):
            print(f"❌ {pol_path} must define 'policy' variable", file=sys.stderr)
            sys.exit(1)
        
        policy_class = pol_mod.policy.__class__.__name__
        if policy_class != 'Policy':
            print(f"❌ Policy class should be named 'Policy', got '{policy_class}'", file=sys.stderr)
            sys.exit(1)
        
        sig = inspect.signature(pol_mod.policy.predict)
        params = list(sig.parameters.keys())
        if 'observation' not in params:
            print("❌ predict() must accept 'observation' parameter", file=sys.stderr)
            sys.exit(1)
        
        print(f"  ✓ Policy validated: {pol_config['name']}")
    
    print("  ✓ Interface test passed")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python validate_single.py <use_case>", file=sys.stderr)
        sys.exit(1)
    
    validate_single(sys.argv[1])