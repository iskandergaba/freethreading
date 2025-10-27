import tomllib

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

author = "Iskander Gaba"
copyright = f"%Y, {author}"
with open("../pyproject.toml", "rb") as f:
    pyproject_data = tomllib.load(f)
    project = pyproject_data["project"]["name"]
    release = pyproject_data["project"]["version"]

# General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
extensions = ["sphinx.ext.napoleon", "sphinx.ext.intersphinx", "autoapi.extension"]

# Options for intersphinx
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Options for AutoAPI
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html

autoapi_dirs = ["../src/freethreading"]
autoapi_type = "python"
autoapi_root = "generated"
autoapi_options = [
    "members",
    "undoc-members",
    "imported-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_member_order = "bysource"
autoapi_keep_files = True
autoapi_add_toctree_entry = False

# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "python_docs_theme"
html_static_path = ["_static"]
html_copy_source = False
html_show_sourcelink = False
html_theme_options = {
    "root_name": "Freethreading",
    "root_url": "https://freethreading.readthedocs.io",
    "issues_url": "https://github.com/iskandergaba/freethreading/issues",
}
