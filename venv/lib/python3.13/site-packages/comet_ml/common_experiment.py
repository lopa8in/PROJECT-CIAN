# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
from typing import Optional

from ._jupyter import display_or_open_browser
from .config import get_backend_address
from .utils import get_root_url


class CommonExperiment(object):
    """
    Class that contains common methods for all experiment types:
        * CometExperiment
        * Experiment
        * OfflineExperiment
        * APIExperiment

    Methods and properties required to use these methods:
        * self.project_id - str or None
        * self._get_experiment_url(tab)
    """

    def display_project(
        self, view_id=None, clear=False, wait=True, new=0, autoraise=True
    ):
        """Show the Comet project page in an IFrame in either (1) a Jupyter
        notebook or Jupyter lab or (2) open a browser window or tab.

        This is only supported for online experiment at the moment.

        Args:
            view_id (str): The id of the view to show.
            clear (bool): To clear the output area, use `clear=True.`
            wait (bool): To wait for the next displayed item, use
                wait=True (cuts down on flashing).
            new (int): Open a new browser window if `new=1`, otherwise
                re-use existing window/tab.
            autoraise (bool): Make the browser tab/window active.

        Raises:
            NotImplementedError: This is not yet supported for offline experiments.

        Note:
            For Jupyter environments, you can utilize the `clear` and `wait` parameters.
            For non-Jupyter environments, you can utilize the `new` and `autoraise` parameters.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start(project_name="comet-docs")

            # Optionally log some metrics or parameters (not required for displaying the project)
            exp.log_metric("accuracy", 0.95)
            exp.log_parameter("learning_rate", 0.01)

            exp.display_project()

            exp.end()
            ```
        """
        if self.project_id is not None:
            server = get_root_url(get_backend_address())
            if view_id is not None:
                url = "%s/api/projects/redirect?projectId=%s?viewId=%s" % (
                    server,
                    self.project_id,
                    view_id,
                )
            else:
                url = "%s/api/projects/redirect?projectId=%s" % (
                    server,
                    self.project_id,
                )

            display_or_open_browser(url, clear, wait, new, autoraise)

    def display(self, clear=False, wait=True, new=0, autoraise=True, tab=None):
        """
        Show the Comet experiment page in an IFrame in a
        Jupyter notebook or Jupyter lab, OR open a browser
        window or tab.

        This is only supported for online experiment at the moment.

        Args:
            clear (bool): To clear the output area, use clear=True. This is only
                used in Notebook environments.
            wait (bool): To wait for the next displayed item, use
                `wait=True` (cuts down on flashing). This is only used in Notebook
                environments.
            new (int): Open a new browser window if `new=1`, otherwise re-use
                existing window/tab. This is only used in non-Notebook
                environments.
            autoraise (bool): Make the browser tab/window active. This is only
                used in non-Notebook environments.
            tab (str): Name of the Tab on Experiment View

        Raises:
            NotImplementedError: This is not yet supported for offline experiments.

        Note:
            The Tab name should be one of:

            * "artifacts"
            * "assets"
            * "audio"
            * "charts"
            * "code"
            * "confusion-matrices"
            * "histograms"
            * "images"
            * "installed-packages"
            * "metrics"
            * "notes"
            * "parameters"
            * "system-metrics"
            * "text"
        """
        url = self._get_experiment_url(tab)
        display_or_open_browser(url, clear, wait, new, autoraise)

    def _get_experiment_url(self, tab: Optional[str] = None):
        raise NotImplementedError()
