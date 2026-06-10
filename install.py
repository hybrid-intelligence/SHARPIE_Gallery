#!/usr/bin/env python
"""
Master installation script for SHARPIE Gallery use cases.

Usage:
    python install.py <use_case>        # Install specific use case
    python install.py --all             # Install all use cases
    python install.py --list            # List available use cases
    python install.py <use_case> --check # Validate without installing
    python install.py --quiet           # Minimal output (errors only)
    python install.py --verbose         # Detailed output
"""

import os
import sys
import subprocess
import argparse
import importlib.util
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent

verbosity_levels = {
    'quiet': 0,
    'default': 1,
    'verbose': 2
}


def parse_args():
    parser = argparse.ArgumentParser(description='Install SHARPIE Gallery use cases')
    parser.add_argument('use_case', nargs='?', help='Use case to install')
    parser.add_argument('--all', action='store_true', help='Install all use cases')
    parser.add_argument('--list', action='store_true', help='List available use cases')
    parser.add_argument('--check', action='store_true', help='Validate without installing')
    parser.add_argument('--sharpie-dir', type=str, default=None,
                        help='Path to SHARPIE directory (default: ../SHARPIE)')
    parser.add_argument('--webserver-dir', type=str, default=None,
                        help='Path to webserver directory (default: <sharpie-dir>/webserver)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output (errors only)')
    parser.add_argument('--verbose', action='store_true', help='Detailed output')
    return parser.parse_args()


def get_paths(args):
    if args.sharpie_dir:
        sharpie_dir = Path(args.sharpie_dir)
    else:
        sharpie_dir = SCRIPT_DIR.parent / 'SHARPIE'
    
    if args.webserver_dir:
        webserver_dir = Path(args.webserver_dir)
    else:
        webserver_dir = sharpie_dir / 'webserver'
    
    return sharpie_dir, webserver_dir


def get_verbosity(args):
    if args.quiet:
        return 0
    if args.verbose:
        return 2
    return 1


def log(msg, level=1, verbosity=1):
    if verbosity >= level:
        print(msg)


def load_config(use_case: str) -> dict:
    config_path = SCRIPT_DIR / use_case / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def list_use_cases():
    use_cases = []
    for item in SCRIPT_DIR.iterdir():
        if item.is_dir() and (item / 'config.yaml').exists():
            use_cases.append(item.name)
    return sorted(use_cases)


def check_dependencies(config: dict, verbosity=1):
    deps = config.get('dependencies', [])
    if not deps:
        log("  No dependencies required", level=2, verbosity=verbosity)
        return
    
    for dep in deps:
        try:
            pkg_name = dep.split('[')[0].replace('-', '_')
            __import__(pkg_name)
            log(f"  ✓ {dep} already installed", level=2, verbosity=verbosity)
        except ImportError:
            log(f"  Installing {dep}...", level=1, verbosity=verbosity)
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])


def validate_files(config: dict, check_only=False, verbosity=1):
    """Validate use case files in an isolated subprocess."""
    use_case = config['use_case']
    
    validate_script = SCRIPT_DIR / 'validate_single.py'
    
    result = subprocess.run(
        [sys.executable, str(validate_script), use_case],
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        for line in result.stdout.rstrip('\n').split('\n'):
            if '✓' in line:
                log(line, level=1, verbosity=verbosity)
    
    if result.returncode != 0:
        error_msg = result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown error"
        if error_msg.startswith('❌ '):
            error_msg = error_msg[3:]
        raise RuntimeError(error_msg)


def setup_database(config: dict, webserver_dir: Path, verbosity=1):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
    sys.path.insert(0, str(webserver_dir))
    import django
    django.setup()
    
    # Django models must be imported after django.setup()
    from experiment.models import Environment, Experiment, Policy, Agent
    
    env_data = config['environment'].copy()
    env, created = Environment.objects.update_or_create(
        name=env_data['name'],
        defaults=env_data
    )
    log(f"  {'Created' if created else 'Updated'} Environment: {env.name}", level=1, verbosity=verbosity)
    
    pol = None
    if 'policy' in config:
        pol_data = config['policy'].copy()
        pol, created = Policy.objects.update_or_create(
            name=pol_data['name'],
            defaults=pol_data
        )
        log(f"  {'Created' if created else 'Updated'} Policy: {pol.name}", level=1, verbosity=verbosity)
    
    created_agents = []
    for agent_config in config['agents']:
        agent_data = agent_config.copy()
        if agent_data.get('policy'):
            agent_data['policy'] = Policy.objects.get(name=agent_data['policy'])
        else:
            agent_data['policy'] = None
        
        agent, created = Agent.objects.update_or_create(
            role=agent_config['role'],
            defaults=agent_data
        )
        created_agents.append(agent)
        log(f"  {'Created' if created else 'Updated'} Agent: {agent.name} ({agent.role})", level=1, verbosity=verbosity)
    
    exp_data = config['experiment'].copy()
    exp_data['environment'] = env
    exp, created = Experiment.objects.update_or_create(
        link=exp_data['link'],
        defaults=exp_data
    )
    
    for agent in created_agents:
        exp.agents.add(agent)
    
    log(f"  {'Created' if created else 'Updated'} Experiment: {exp.name} (link: {exp.link})", level=1, verbosity=verbosity)


def show_installation_notes(config: dict, verbosity=1):
    notes = config.get('installation_notes')
    if notes:
        log("\n📝 Installation Notes:", level=1, verbosity=verbosity)
        log(f"   {notes}\n", level=1, verbosity=verbosity)


def install_use_case(use_case: str, webserver_dir: Path, check_only=False, verbosity=1):
    action = 'Validating' if check_only else 'Installing'
    log(f"\n{action} {use_case}...", level=1, verbosity=verbosity)
    
    config = load_config(use_case)
    
    show_installation_notes(config, verbosity)
    
    log("Step 1/3: Checking dependencies...", level=1, verbosity=verbosity)
    check_dependencies(config, verbosity)
    
    log("\nStep 2/3: Validating files...", level=1, verbosity=verbosity)
    validate_files(config, check_only=check_only, verbosity=verbosity)
    
    if check_only:
        return
    
    log("\nStep 3/3: Setting up database...", level=1, verbosity=verbosity)
    setup_database(config, webserver_dir, verbosity)
    
    log(f"\n✅ {use_case} installed successfully!", level=1, verbosity=verbosity)


def main():
    args = parse_args()
    verbosity = get_verbosity(args)
    _, webserver_dir = get_paths(args)
    
    if args.list:
        log("Available use cases:", level=1, verbosity=verbosity)
        for uc in list_use_cases():
            log(f"  - {uc}", level=1, verbosity=verbosity)
        return
    
    if args.all:
        failed = []
        for uc in list_use_cases():
            try:
                install_use_case(uc, webserver_dir, check_only=args.check, verbosity=verbosity)
            except Exception as e:
                log(f"❌ {uc} failed: {e}", level=0, verbosity=verbosity)
                failed.append(uc)
                continue
        
        if failed:
            log(f"\n❌ {len(failed)} use case(s) failed validation", level=0, verbosity=verbosity)
            sys.exit(1)
    elif args.use_case:
        install_use_case(args.use_case, webserver_dir, check_only=args.check, verbosity=verbosity)
    else:
        import argparse
        argparse.ArgumentParser(description='Install SHARPIE Gallery use cases').print_help()


if __name__ == '__main__':
    main()