from functools import cached_property

from pydantic import DirectoryPath, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from antarctica_today.constants.paths import REPO_DIR


class Settings(BaseSettings):
    """Configuration required to download brightness temperature data."""

    model_config = SettingsConfigDict(
        env_prefix="ANTARCTICA_TODAY_",
        env_file=".env",
        extra="ignore",
    )

    # New data directories:
    # - Keep data included in the repo separate from runtime data. We don't want to require
    #   a specific directory structure that may not work on every computer. For example, on
    #   NSIDC VMs, we have limited direct storage, and need to use mounts to access larger
    #   storage devices.
    # - Use environment variables to enable override; in this case I think we only need one
    #   for the root storage directory. Default to an in-repo storage location so if the
    #   envvars are not populated, system pollution doesn't occur.
    # - Migrate more things iteratively :)
    STORAGE_BASEDIR: DirectoryPath = REPO_DIR

    @computed_field  # type:ignore[misc]
    @cached_property
    def db_dir(self) -> DirectoryPath:
        directory = self.STORAGE_BASEDIR / "database"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @computed_field  # type:ignore[misc]
    @cached_property
    def plots_dir(self) -> DirectoryPath:
        directory = self.STORAGE_BASEDIR / "plots"
        directory.mkdir(parents=True, exist_ok=True)
        return directory
