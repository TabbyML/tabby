import inspect
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
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
]

source_suffix = [".rst", ".md"]
exclude_patterns = ["README.md"]

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

autosectionlabel_prefix_document = True

napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True


def fix_pybind11_signatures(
    app, what, name, obj, options, signature, return_annotation
):
    def _remove_self(signature):
        arguments = signature[1:-1].split(", ")
        if arguments and arguments[0].startswith("self:"):
            arguments.pop(0)
        return "(%s)" % ", ".join(arguments)

    def _reformat_typehints(content):
        return content.replace(
            "ctranslate2._ext.",
            "ctranslate2." if autodoc_typehints_format == "fully-qualified" else "",
        )

    if signature is not None:
        signature = _remove_self(signature)
        signature = _reformat_typehints(signature)

    if return_annotation is not None:
        return_annotation = _reformat_typehints(return_annotation)

    return (signature, return_annotation)


def skip_pybind11_builtin_members(app, what, name, obj, skip, options):
    skipped_entries = {
        "__init__": ["self", "args", "kwargs"],
        "__new__": ["args", "kwargs"],
    }

    ref_arguments = skipped_entries.get(name)
    if ref_arguments is not None:
        try:
            arguments = list(inspect.signature(obj).parameters.keys())
            if arguments == ref_arguments:
                return True
        except ValueError:
            pass

    return None


def setup(app):
    app.add_css_file("custom.css")
    app.connect("autodoc-skip-member", skip_pybind11_builtin_members)
    app.connect("autodoc-process-signature", fix_pybind11_signatures)
