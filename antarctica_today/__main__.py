import click
from loguru import logger

from antarctica_today.generate_daily_melt_file import generate_new_daily_melt_files
from antarctica_today.generate_gap_filled_melt_picklefile import (
    save_gap_filled_picklefile,
)
from antarctica_today.main import (
    generate_all_plots_and_maps_main,
    preprocessing_main,
)
from antarctica_today.melt_array_picklefile import save_model_array_picklefile
from antarctica_today.nsidc_download_Tb_data import download_new_files
from antarctica_today.update_data import update_everything_to_latest_date


@click.group()
def cli():
    """Antarctica Today."""
    pass


@cli.group()
def init():
    """Initialize the Antarctica Today database."""
    pass


@init.command("01-download-tb")
def download_tb():
    """Download NSIDC-0080 brightness temperature granules.

    By default, data is downloaded starting on 2022-01-10, as data before then is
    committed to this repository. This pre-generated data was generated from NSIDC-0001
    and NSIDC-0007 datasets.
    """
    download_new_files()
    logger.success("Download of NSIDC-0080 granules complete.")


@init.command("02-generate-daily-melt")
def generate_daily_melt():
    """Generate daily melt files from brightness temperature granules.

    These files are written as binary files with ".bin" extension in the
    `/data/daily_melt_bin_files/` directory in this repository.
    """
    generate_new_daily_melt_files(overwrite=False)
    logger.success("Generation of daily melt files complete.")


@init.command("03-preprocess")
def preprocess():
    """Perform pre-processing steps for Antarctica Today data.

    This includes CSV, TIF, and pickled numpy arrays with ".pickle" extension.
    """
    preprocessing_main()
    logger.success("Preprocess complete.")


@init.command("all")
def init_all():
    """Initialize the Antarctica Today database."""
    download_new_files()
    generate_new_daily_melt_files(overwrite=False)
    preprocessing_main()
    logger.success("Antarctica Today database initialized.")


@cli.command()
def daily_update_and_plots():
    """Perform a daily update of the Antarctica Today database and produce plots.

    This includes:

    * A "daily melt" map of the most recent day's melt extent
    * A "sum" map of the current season's total melt days
    * An "anomaly" map of that season's total melt days in comparison to baseline
      average values to-that-day-of-year
    * A line plot of melt extent up do that date, compared to historical baseline
      averages

    It will copy these plots into a sub-directory `/plots/daily_plots_gathered/[date]/`
    for easy collection.
    """
    update_everything_to_latest_date(copy_to_gathered_dir=True)
    logger.success("Database updated to the current date. New plots produced.")


@cli.command()
def all_plots():
    """Generate all Antarctica Today plot images from the database."""
    generate_all_plots_and_maps_main()
    logger.success("All plots generated.")


if __name__ == "__main__":
    cli()
