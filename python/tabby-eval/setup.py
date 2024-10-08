from setuptools import find_packages, setup

setup(
    name="tabby_data_pipeline",
    packages=find_packages(exclude=["tabby_data_pipeline_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud",
        "dagstermill",
        "papermill-origami>=0.0.8",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
