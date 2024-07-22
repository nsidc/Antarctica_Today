import click

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


@click.group()
def cli():
    """Antarctica Today."""
    pass


@click.option(
    "--start-date",
    default="2022-01-10",
    help="TODO",
)
@cli.command()
def download_tb(start_date: str):
    """Download NSIDC-0080 brightness temperature granules.

    The default start date is the day after the end of the .bin data available in
    `/data/daily_melt_bin_files/` directory in this repo.
    """
    download_new_files(time_start=start_date)


@click.option(
    "--start-date",
    # TODO: Is this default right? The actual function we're calling has a default of
    # 2021-10-01. Why?
    default="2022-01-10",
    help="TODO",
)
@cli.command()
def generate_daily_melt(start_date: str):
    """Generate daily melt file from brightness temperature granules."""
    generate_new_daily_melt_files(
        start_date=start_date,
        overwrite=False,
    )


@cli.command
def melt_array_picklefile():
    """Is this needed operationally?"""
    save_model_array_picklefile()


@cli.command
def gap_filled_melt_picklefile():
    """Is this needed operationally?"""
    save_gap_filled_picklefile()


@cli.command()
def preprocess():
    """Perform pre-processing steps for Antarctica Today data.

    This includes:
        - ...
        - ...
    """
    preprocessing_main()


@cli.command()
def process():
    """Perform processing steps for Antarctica Today data.

    This includes:
        - ...
        - ...
    """
    generate_all_plots_and_maps_main()


if __name__ == "__main__":
    cli()
