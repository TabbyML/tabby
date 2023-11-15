from dagster import Definitions, load_assets_from_modules

from dagstermill import define_dagstermill_asset, ConfigurableLocalOutputNotebookIOManager

from dagster import AssetIn, Field, Int, asset, file_relative_path

from . import assets

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    resources = {
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager()
    }
)


