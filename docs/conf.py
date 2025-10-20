"""Configuration file for the Sphinx documentation builder."""

import sys
from pathlib import Path

import tomllib

# Add the project source to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Freethreading"
copyright = "2025, Iskander Gaba"
author = "Iskander Gaba"
with open("../pyproject.toml", "rb") as f:
    pyproject_data = tomllib.load(f)
    release = pyproject_data["project"]["version"]

# General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

# Autosummary settings
autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Options for autodoc
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "class"
autodoc_inherit_docstrings = False

# Options for intersphinx
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Freethreading"
html_theme_options = {
    "source_repository": "https://github.com/iskandergaba/freethreading",
    "source_branch": "main",
    "source_directory": "docs/",
}
