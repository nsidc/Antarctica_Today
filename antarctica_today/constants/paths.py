import os
from pathlib import Path
from typing import Final

_this_dir: Final = Path(__file__).parent

# The directory containing the main Python package:
PACKAGE_DIR: Final = _this_dir.parent

# The root of the Git repository (directory containing `.git`)
REPO_DIR: Final = PACKAGE_DIR.parent

# New data directories:
# - Keep data included in the repo separate from runtime data. We don't want to require
#   a specific directory structure that may not work on every computer. For example, on
#   NSIDC VMs, we have limited direct storage, and need to use mounts to access larger
#   storage devices.
# - Use environment variables to enable override; in this case I think we only need one
#   for the root storage directory. Default to an in-repo storage location so if the
#   envvars are not populated, system pollution doesn't occur.
# - Migrate more things iteratively :)
_default_storage_dir = REPO_DIR
STORAGE_DIR: Final = Path(os.environ.get("ANTARCTICA_TODAY_STORAGE_DIR", _default_storage_dir))
DATA_DATABASE_DIR: Final = STORAGE_DIR / "database"
# DATA_OUTPUT_DIR: Final = STORAGE_DIR / "output"

# Legacy data directories
DATA_DIR: Final = REPO_DIR / "data"
DATA_QGIS_DIR: Final = REPO_DIR / "qgis"
DATA_TB_DIR: Final = REPO_DIR / "Tb"
DATA_PLOTS_DIR: Final = REPO_DIR / "plots"
DATA_BASELINE_DATASETS_DIR: Final = REPO_DIR / "baseline_datasets"


# TODO: Bring tb_file_data.py into constants
