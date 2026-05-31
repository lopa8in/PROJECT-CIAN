# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import logging
from typing import Any, Dict

import requests

LOGGER = logging.getLogger(__name__)
DEFAULT_JUPYTER_INTERACTIVE_FILE_NAME = "Jupyter interactive"
DEFAULT_JUPYTER_CODE_ASSET_NAME = "Code.ipynb"
DEFAULT_COLAB_NOTEBOOK_ASSET_NAME = "ColabNotebook.ipynb"


def _in_jupyter_environment() -> bool:
    """
    Check to see if code is running in a Jupyter environment,
    including jupyter notebook, lab, or console.
    """
    try:
        import IPython
    except Exception:
        return False

    ipy = IPython.get_ipython()
    if ipy is None or not hasattr(ipy, "kernel"):
        return False
    else:
        return True


def _in_ipython_environment() -> bool:
    """
    Check to see if code is running in an IPython environment.
    """
    try:
        import IPython
    except Exception:
        return False

    ipy = IPython.get_ipython()
    if ipy is None:
        return False
    else:
        return True


def _in_colab_environment() -> bool:
    """
    Check to see if code is running in Google colab.
    """
    try:
        import IPython
    except Exception:
        return False

    ipy = IPython.get_ipython()
    return "google.colab" in str(ipy)


def _get_colab_session_api_url() -> str:
    """For manual IP check use the following inside Colab notebook:
        # Gives IP address
        !hostname -I
        print()  # empty line
        # Gives IP addresses with port numbers
        !sudo lsof -i -P -n | grep LISTEN
    OR
        colab_ip = %system hostname -I   # uses colab magic to get list from bash
        colab_ip = colab_ip[0].strip()   # returns "172.28.0.12"
        colab_port = 9000                # could use 6000, 8080, or 9000

    https://stackoverflow.com/questions/61901628/it-is-possible-to-assign-the-google-colab-notebook-name-to-a-python-variable-in
    """
    from socket import gethostbyname, gethostname

    ip = gethostbyname(gethostname())
    return f"http://{ip}:9000/api/sessions"  # could use any of the ports: 6000, 8080, or 9000


def _get_colab_notebook_url() -> str:
    response = requests.get(_get_colab_session_api_url(), timeout=1)
    response.raise_for_status()
    notebook_file_id = response.json()[0]["path"]
    return "https://colab.research.google.com/notebook#%s" % notebook_file_id


def _get_colab_notebook_json() -> Dict[str, Any]:
    import google.colab._message

    notebook_json = google.colab._message.blocking_request(request_type="get_ipynb")
    return notebook_json["ipynb"]


def display_or_open_browser(
    url: str,
    clear: bool = False,
    wait: bool = True,
    new: int = 0,
    autoraise: bool = True,
) -> None:
    if _in_jupyter_environment():
        from IPython.display import IFrame, clear_output, display

        if clear:
            clear_output(wait=wait)
        display(IFrame(src=url, width="100%", height="800px"))
    else:
        import webbrowser

        webbrowser.open(url, new=new, autoraise=autoraise)
