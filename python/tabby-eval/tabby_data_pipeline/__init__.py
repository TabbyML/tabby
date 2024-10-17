from dagster import Definitions, load_assets_from_modules

from dagstermill import ConfigurableLocalOutputNotebookIOManager


from . import assets, create_csv

all_assets = load_assets_from_modules([assets, create_csv])

defs = Definitions(
    assets=all_assets,
    resources = {
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager()
    }
)


