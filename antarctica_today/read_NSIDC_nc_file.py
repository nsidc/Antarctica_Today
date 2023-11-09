from pathlib import Path

import numpy as np
import xarray as xra

from antarctica_today.constants.grid import DEFAULT_GRID_SHAPE
from antarctica_today.constants.paths import DATA_TB_DIR

def read_NSIDC_nc_file(fname: Path) -> dict[str, np.ndarray]:
    """Read an NSIDC-0080v2 file and return a 2D data array for each desired channel.

    NOTE: Unlike `read_NSIDC_bin_file`, we don't need to worry about the scaling
    multiplier. It's baked in to the NetCDF metadata (see
    `ds.TB_F18_SH_19V.encoding["scale_factor"]`) and Xarray automatically scales on
    read.
    """
    # TODO: Which satellite? Why?
    SATELLITE = "F18"

    ds = xra.open_dataset(fname, group=SATELLITE)

    # Drop the first dimension; its size should be 1 and the rest of the code expects
    # 2D, not 3D.
    channel_data = {
        "Tb_array_37h": np.squeeze(ds.TB_F18_SH_37H.data),
        "Tb_array_37v": np.squeeze(ds.TB_F18_SH_37V.data),
        "Tb_array_19v": np.squeeze(ds.TB_F18_SH_19V.data),
    }

    for arrname, ndarray in channel_data.items():
        shape = ndarray.shape
        if shape != DEFAULT_GRID_SHAPE:
            raise RuntimeError(
                f"Unexpected grid shape {shape} in {str(fname)}:{arr_name[-3:]}."
                f" Expected {DEFAULT_GRID_SHAPE}."
            )

    return channel_data
