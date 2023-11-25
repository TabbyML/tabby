import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="tabby-python-client",
    version="0.4.0-dev",
    description="A client library for accessing Tabby Server",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8, <4",
    install_requires=["httpx >= 0.15.0, < 0.25.0", "attrs >= 21.3.0", "python-dateutil >= 2.8.0, < 3"],
    package_data={"tabby_python_client": ["py.typed"]},
)
