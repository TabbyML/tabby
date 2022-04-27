import os
import sys

import ctranslate2

# General information about the project.
project = "CTranslate2"
author = "OpenNMT"
language = "en"

release = ctranslate2.__version__  # The full version, including alpha/beta/rc tags.
version = ".".join(release.split(".")[:2])  # The short X.Y version.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
]

source_suffix = [".rst"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {}
html_static_path = ["_static"]
html_show_sourcelink = False
html_show_copyright = False
html_show_sphinx = False
html_favicon = "_static/favicon.png"

autodoc_member_order = "groupwise"
autodoc_typehints_format = "short"
autodoc_class_signature = "separated"
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True

def setup(app):
    app.add_css_file("custom.css")
