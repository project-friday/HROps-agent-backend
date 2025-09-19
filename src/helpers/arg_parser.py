import sys
from typing import List, Tuple


def extract_arg(argv: List[str], flag: str, default: str) -> Tuple[str, List[str]]:
    """
    Extract a CLI argument value for `flag`, returning (value, cleaned_argv).
    If the flag is not found, returns (default, argv).
    """
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            value = argv[idx + 1]
            # strip both flag and its value from argv
            argv = argv[:idx] + argv[idx + 2 :]
            print(f"Using {flag.lstrip('-')}: {value}")
            return value, argv
    return default, argv


def parse_cli_args(argv: List[str]) -> Tuple[str, str, List[str]]:
    """Extract --tenant and --flow from argv, stripping them to avoid CLI conflicts."""
    tenant, argv = extract_arg(argv, "--tenant", "walmart")
    flow, argv = extract_arg(argv, "--flow", "default")
    return tenant, flow, argv
