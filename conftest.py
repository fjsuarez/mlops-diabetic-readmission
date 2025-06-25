import sys
from pathlib import Path

# Add both project root and src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"

# Add project root first (for src.module imports)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src path (for direct module imports)
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
