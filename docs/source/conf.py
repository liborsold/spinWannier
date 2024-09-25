# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# INCLUDE package in path 

import os 
import sys
sys.path.insert(0, os.path.abspath('../../src/'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'spinWannier'
copyright = '2024, Libor Vojáček'
author = 'Libor Vojáček'
release = '0.1.6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton'
] 

templates_path = ['_templates']
html_static_path = ['_static']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "repository_url": "https://github.com/liborsold/spinWannier",
    "use_source_button": False,
    "use_repository_button": True,
    "repository_branch": "master",
    "path_to_docs": "docs/source/",
    "logo": {
        "text": f"<b>spinWannier</b> {release} documentation", # "image_light": "_static/logo-light.png","image_dark": "_static/logo-light.png",
    },
}

html_favicon = '_static/favicon/android-chrome-192x192.png'

