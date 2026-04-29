#!/usr/bin/env python
"""
Master installation script for SHARPIE Gallery use cases.

Usage:
    python install.py <use_case>        # Install specific use case
    python install.py --all             # Install all use cases
    python install.py --list            # List available use cases
    python install.py <use_case> --check # Validate without installing
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import argparse

SCRIPT_DIR = Path(__file__).parent
SHARPIE_DIR = SCRIPT_DIR.parent / 'SHARPIE'
WEBSERVER_DIR = SHARPIE_DIR / 'webserver'

def load_config(use_case: str) -> dict:
    """Load config.json for a use case"""
    config_path = SCRIPT_DIR / use_case / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        return json.load(f)

def list_use_cases():
    """List all available use cases"""
    use_cases = []
    for item in SCRIPT_DIR.iterdir():
        if item.is_dir() and (item / 'config.json').exists():
            use_cases.append(item.name)
    return sorted(use_cases)

def check_dependencies(config: dict):
    """Install required dependencies"""
    deps = config.get('dependencies', [])
    for dep in deps:
        try:
            pkg_name = dep.split('[')[0].replace('-', '_')
            __import__(pkg_name)
            print(f"  ✓ {dep} already installed")
        except ImportError:
            print(f"  Installing {dep}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])

def validate_files(config: dict, check_only=False):
    """Validate environment.py and policy.py"""
    use_case = config['use_case']
    use_case_dir = SCRIPT_DIR / use_case
    
    # Validate environment
    env_config = config['environment']
    env_file = env_config['filepaths']['environment'].split('/')[-1]
    env_path = use_case_dir / env_file
    
    if not env_path.exists():
        raise FileNotFoundError(f"Missing environment: {env_path}")
    
    # Import and validate environment
    import importlib.util
    spec = importlib.util.spec_from_file_location("environment", env_path)
    env_mod = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(env_mod)
    except Exception as e:
        raise RuntimeError(f"Failed to load {env_path}: {e}")
    
    if not hasattr(env_mod, 'environment'):
        raise AttributeError(f"{env_path} must define 'environment' variable")
    
    env = env_mod.environment
    required_methods = ['reset', 'step', 'render']
    for method in required_methods:
        if not hasattr(env, method):
            raise AttributeError(f"environment must have {method}() method")
    
    print(f"  ✓ Environment validated: {env_config['name']}")
    
    # Validate policy (if exists)
    if 'policy' in config:
        pol_config = config['policy']
        
        # Check all required files exist
        for key, filepath in pol_config['filepaths'].items():
            pol_file = filepath.split('/')[-1]
            pol_path = use_case_dir / pol_file
            if not pol_path.exists():
                raise FileNotFoundError(f"Missing {key}: {pol_path}")
        
        # Validate main policy file
        pol_file = pol_config['filepaths']['policy'].split('/')[-1]
        pol_path = use_case_dir / pol_file
        
        spec = importlib.util.spec_from_file_location("policy", pol_path)
        pol_mod = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(pol_mod)
        except Exception as e:
            raise RuntimeError(f"Failed to load {pol_path}: {e}")
        
        if not hasattr(pol_mod, 'policy'):
            raise AttributeError(f"{pol_path} must define 'policy' variable")
        
        # Validate policy class name
        policy_class = pol_mod.policy.__class__.__name__
        if policy_class != 'Policy':
            raise ValueError(f"Policy class should be named 'Policy', got '{policy_class}'")
        
        # Validate predict signature
        import inspect
        sig = inspect.signature(pol_mod.policy.predict)
        params = list(sig.parameters.keys())
        if 'observation' not in params:
            raise ValueError("predict() must accept 'observation' parameter")
        
        print(f"  ✓ Policy validated: {pol_config['name']}")
    
    if check_only:
        return
    
    # Test basic interface
    print("  Testing interface...")
    obs, info = env.reset()
    if 'policy' in config:
        action = pol_mod.policy.predict(observation=obs)
        agent_role = config['agents'][0]['role']
        result = env.step({agent_role: action})
        if len(result) != 5:
            raise ValueError("step() must return 5 values (obs, reward, terminated, truncated, info)")
    
    print(f"  ✓ Interface test passed")

def setup_database(config: dict):
    """Setup database entries from config"""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webserver.settings')
    sys.path.insert(0, str(WEBSERVER_DIR))
    import django
    django.setup()
    
    from experiment.models import Environment, Experiment, Policy, Agent
    
    # Create Environment
    env_data = config['environment'].copy()
    env, created = Environment.objects.update_or_create(
        name=env_data['name'],
        defaults=env_data
    )
    print(f"  {'Created' if created else 'Updated'} Environment: {env.name}")
    
    # Create Policy (if exists)
    pol = None
    if 'policy' in config:
        pol_data = config['policy'].copy()
        pol, created = Policy.objects.update_or_create(
            name=pol_data['name'],
            defaults=pol_data
        )
        print(f"  {'Created' if created else 'Updated'} Policy: {pol.name}")
    
    # Create Agents
    created_agents = []
    for agent_config in config['agents']:
        agent_data = agent_config.copy()
        # Resolve policy reference
        if agent_data.get('policy'):
            agent_data['policy'] = Policy.objects.get(name=agent_data['policy'])
        else:
            agent_data['policy'] = None
        
        agent, created = Agent.objects.update_or_create(
            role=agent_config['role'],
            defaults=agent_data
        )
        created_agents.append(agent)
        print(f"  {'Created' if created else 'Updated'} Agent: {agent.name} ({agent.role})")
    
    # Create Experiment
    exp_data = config['experiment'].copy()
    exp_data['environment'] = env
    exp, created = Experiment.objects.update_or_create(
        link=exp_data['link'],
        defaults=exp_data
    )
    
    # Add agents
    for agent in created_agents:
        exp.agents.add(agent)
    
    print(f"  {'Created' if created else 'Updated'} Experiment: {exp.name} (link: {exp.link})")

def show_installation_notes(config: dict):
    """Show special installation notes"""
    notes = config.get('installation_notes')
    if notes:
        print("\n📝 Installation Notes:")
        print(f"   {notes}\n")

def install_use_case(use_case: str, check_only=False):
    """Install a single use case"""
    print(f"\n{'Validating' if check_only else 'Installing'} {use_case}...")
    
    config = load_config(use_case)
    
    show_installation_notes(config)
    
    print("Step 1/3: Checking dependencies...")
    check_dependencies(config)
    
    print("\nStep 2/3: Validating files...")
    validate_files(config, check_only=check_only)
    
    if check_only:
        return
    
    print("\nStep 3/3: Setting up database...")
    setup_database(config)
    
    print(f"\n✅ {use_case} installed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Install SHARPIE Gallery use cases')
    parser.add_argument('use_case', nargs='?', help='Use case to install')
    parser.add_argument('--all', action='store_true', help='Install all use cases')
    parser.add_argument('--list', action='store_true', help='List available use cases')
    parser.add_argument('--check', action='store_true', help='Validate without installing')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available use cases:")
        for uc in list_use_cases():
            print(f"  - {uc}")
        return
    
    if args.all:
        failed = []
        for uc in list_use_cases():
            try:
                install_use_case(uc, check_only=args.check)
            except Exception as e:
                print(f"❌ {uc} failed: {e}")
                failed.append(uc)
                continue
        
        if failed:
            print(f"\n❌ {len(failed)} use case(s) failed validation")
            sys.exit(1)
    elif args.use_case:
        install_use_case(args.use_case, check_only=args.check)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()