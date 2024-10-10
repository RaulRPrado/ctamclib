"""Configuration files for the Sphinx documentation build for simtools."""

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# pylint: skip-file
import os
import sys
from pathlib import Path

import toml
import yaml

import simtools.version

sys.path.insert(0, os.path.abspath("../../simtools"))
sys.path.insert(0, os.path.abspath("../../simtools/applications"))
sys.path.insert(0, os.path.abspath("../.."))


def get_authors_from_citation_file():
    """Read list of authors from CITATION.cff file."""
    with open(Path(__file__).parent / "../../CITATION.cff", encoding="utf-8") as file:
        citation = yaml.safe_load(file)

    tmp_author = ""
    try:
        for person in citation["authors"]:
            tmp_author = tmp_author + person["given-names"] + " " + person["family-names"]
            tmp_author += " (" + person["affiliation"] + "), "
    except KeyError:
        pass
    return tmp_author[:-2]


def get_python_version_from_pyproject():
    """Read python version from pyproject.toml file."""
    with open(Path(__file__).parent / "../../pyproject.toml") as file:
        pyproject = toml.load(file)

    return (
        pyproject["project"]["requires-python"],
        pyproject["project"]["requires-python"].replace(">", "").replace("=", ""),
    )


# -- Project information -----------------------------------------------------

project = "simtools"
copyright = "2024, gammasim-tools, simtools developers"  # noqa A001
author = get_authors_from_citation_file()
rst_epilog = f"""
.. |author| replace:: {author}
"""

python_min_requires, python_requires = get_python_version_from_pyproject()
rst_epilog = f"""
.. |python_min_requires| replace:: {python_min_requires}
"""

myst_substitutions = {
    "python_min_requires": python_min_requires,
}

# The short X.Y version
version = str(simtools.version.__version__)
# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
    "numpydoc",
]

# Display todos by setting to True
todo_include_todos = True

# Change the look of autodoc classes
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst"]

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = "sphinx"
default_role = "py:obj"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

html_title = f"{project} v{version} Manual"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "path_to_docs": "docs/source",
    "repository_url": "https://github.com/gammasim/simtools",
    "repository_branch": "main",
    "use_issues_button": True,
    "show_toc_level": 1,
    "announcement": (
        "simtools is under rapid development with continuous changes. "
        "Please contact the developers before using it."
    ),
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/gammasim/simtools",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/gammasimtools/",
            "icon": "https://badge.fury.io/py/gammasimtools.svg",
            "type": "url",
        },
    ],
    "home_page_in_toc": True,
    "use_source_button": True,
    "use_download_button": True,
    "navigation_with_keys": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Output file base name for HTML help builder.
htmlhelp_basename = "simtoolsdoc"

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": (f"https://docs.python.org/{python_requires}", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "astropy": ("https://docs.astropy.org/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# myst (markdown options)
myst_heading_anchors = 3
myst_enable_extensions = {
    "colon_fence",
    "substitution",
}

suppress_warnings = ["myst.*", "myst.duplicate_def"]
