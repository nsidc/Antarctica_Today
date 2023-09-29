from pathlib import Path
from typing import Final

_this_dir: Final = Path(__file__).parent

# The directory containing the main Python package:
PACKAGE_DIR: Final = _this_dir.parent

# The root of the Git repository (directory containing `.git`)
REPO_DIR: Final = PACKAGE_DIR.parent

DATA_DIR: Final = REPO_DIR / "data"
DATA_QGIS_DIR: Final = REPO_DIR / "qgis"
DATA_TB_DIR: Final = REPO_DIR / "Tb"
DATA_PLOTS_DIR: Final = REPO_DIR / "plots"
DATA_BASELINE_DATASETS_DIR: Final = REPO_DIR / "baseline_datasets"

# TODO: Bring tb_file_data.py into constants
