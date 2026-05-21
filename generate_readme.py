#!/usr/bin/env python
"""Generate README.md from config.yaml for each use case"""

import yaml
from pathlib import Path


def generate_readme(config: dict) -> str:
    uc = config['use_case']
    env = config['environment']
    exp = config['experiment']
    
    lines = [
        f"# {exp['name']}",
        "",
        exp['long_description'],
        "",
        "## Installation",
        "",
        "From the gallery root:",
        "```bash",
        f"python install.py {uc}",
        "```",
        "",
    ]
    
    if config.get('installation_notes'):
        lines.extend([
            "### ⚠️ Important",
            "",
            config['installation_notes'],
            "",
        ])
    
    lines.extend([
        "## Dependencies",
        "",
    ])
    
    deps = config.get('dependencies', [])
    if deps:
        for dep in deps:
            lines.append(f"- {dep}")
    else:
        lines.append("No additional dependencies required.")
    
    lines.extend([
        "",
        "## Configuration",
        "",
        "This use case has the following agents:",
        "",
    ])
    
    for agent in config['agents']:
        inputs_desc = "human" if agent['participant'] else "AI"
        policy_desc = f" (policy: {agent['policy']})" if agent.get('policy') else ""
        lines.append(f"- **{agent['name']}** ({agent['role']}): {inputs_desc} inputs{policy_desc}")
        
        if agent['participant'] and agent.get('keyboard_input_display'):
            lines.append("  - Keyboard controls:")
            for key, cfg in agent['keyboard_input_display'].items():
                symbol = cfg.get('symbol', key)
                label = cfg.get('label', key)
                lines.append(f"    - {symbol} ({label})")
    
    lines.extend([
        "",
        "See `config.yaml` for full configuration details.",
    ])
    
    return '\n'.join(lines)


def main():
    script_dir = Path(__file__).parent
    use_cases = [d for d in script_dir.iterdir() if d.is_dir() and (d / 'config.yaml').exists()]
    
    for uc_dir in use_cases:
        config_path = uc_dir / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        readme_content = generate_readme(config)
        readme_path = uc_dir / 'README.md'
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"Generated {readme_path}")


if __name__ == '__main__':
    main()