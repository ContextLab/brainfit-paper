"""
Config file read by the notebook server with some options that presets
running notebooks inside a container easier
"""
from os import getenv


c.NotebookApp.ip = getenv("NOTEBOOK_IP")
c.NotebookApp.port = int(getenv("NOTEBOOK_PORT"))
c.NotebookApp.notebook_dir = getenv("NOTEBOOK_DIR")
c.NotebookApp.open_browser = False
c.NotebookApp.allow_root = True
# https://github.com/jupyter/notebook/issues/3130
c.FileContentsManager.delete_to_trash = False
