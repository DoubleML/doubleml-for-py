"""
Script to systematically remove optuna_settings from all _nuisance_tuning methods
and _dml_tune calls across the DoubleML codebase.
"""

import re
from pathlib import Path

def fix_nuisance_tuning_signature(content):
    """Remove optuna_settings=None from _nuisance_tuning method signatures."""
    # Pattern: method signature with optuna_settings parameter
    pattern = r'(def _nuisance_tuning.*?n_iter_randomized_search,)\s*optuna_settings=None,'
    replacement = r'\1'
    return re.sub(pattern, replacement, content, flags=re.DOTALL)

def fix_dml_tune_calls(content):
    """Remove optuna_settings argument from _dml_tune function calls."""
    # Pattern: _dml_tune call with optuna_settings argument
    pattern = r'(_dml_tune\([^)]+?n_iter_randomized_search,)\s*optuna_settings,\s*\n(\s*learner_name=)'
    replacement = r'\1\n\2'
    return re.sub(pattern, replacement, content, flags=re.DOTALL)

def process_file(file_path):
    """Process a single file to remove optuna_settings references."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        content = fix_nuisance_tuning_signature(content)
        content = fix_dml_tune_calls(content)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Updated"
        return False, "No changes needed"
    except Exception as e:
        return False, f"Error: {str(e)}"

# Files to process
files_to_update = [
    r'c:\Users\Work\Documents\GitHub\doubleml-for-py\doubleml\did\did.py',
    r'c:\Users\Work\Documents\GitHub\doubleml-for-py\doubleml\did\did_cs.py',
    r'c:\Users\Work\Documents\GitHub\doubleml-for-py\doubleml\did\did_cs_binary.py',
    r'c:\Users\Work\Documents\GitHub\doubleml-for-py\doubleml\irm\apo.py',
    r'c:\Users\Work\Documents\GitHub\doubleml-for-py\doubleml\irm\cvar.py',
    r'c:\Users\Work\Documents\GitHub\doubleml-for-py\doubleml\irm\iivm.py',
    r'c:\Users\Work\Documents\GitHub\doubleml-for-py\doubleml\irm\irm.py',
    r'c:\Users\Work\Documents\GitHub\doubleml-for-py\doubleml\irm\lpq.py',
    r'c:\Users\Work\Documents\GitHub\doubleml-for-py\doubleml\irm\pq.py',
    r'c:\Users\Work\Documents\GitHub\doubleml-for-py\doubleml\irm\ssm.py',
    r'c:\Users\Work\Documents\GitHub\doubleml-for-py\doubleml\plm\pliv.py',
    r'c:\Users\Work\Documents\GitHub\doubleml-for-py\doubleml\tests\test_nonlinear_score_mixin.py',
]

print("=" * 80)
print("Fixing optuna_settings references across DoubleML codebase")
print("=" * 80)

results = []
for file_path in files_to_update:
    changed, status = process_file(file_path)
    file_name = Path(file_path).name
    results.append((file_name, changed, status))
    print(f"{'✓' if changed else '○'} {file_name:30s} - {status}")

print("\n" + "=" * 80)
print(f"Summary: {sum(1 for _, changed, _ in results if changed)} files updated")
print("=" * 80)
