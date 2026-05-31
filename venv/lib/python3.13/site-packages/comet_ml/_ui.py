# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import functools
import logging
import types

LOGGER = logging.getLogger(__name__)


def _get_object_type(obj):
    name = ""
    if isinstance(obj, types.ModuleType):
        name += obj.__name__ + "."
    if hasattr(obj, "__module__"):
        name += obj.__module__ + "."
    if hasattr(obj, "__class__"):
        name += obj.__class__.__name__
    return name


class UI:
    def __init__(self, column=None):
        """
        Main UI for Comet's python panel widgets.
        """
        self._column = column

    @property
    def _root(self):
        if self._column is not None:
            return self._column
        else:
            return self._st

    @property
    @functools.lru_cache()
    def _st(cls):
        try:
            import streamlit as st

            return st
        except ImportError:
            raise Exception(
                "Comet Python Panels requires running inside a streamlit app"
            )

    @property
    def session_state(self):
        """
        Return the underlying session state.
        """
        return self._st.session_state

    def set_css(self, css):
        """
        This method has been deprecated.
        """
        self._root.markdown(
            "<style>{css}</style>".format(css=css), unsafe_allow_html=True
        )

    def add_css(self, css):
        """
        This method has been deprecated.
        """
        self._root.markdown(
            "<style>{css}</style>".format(css=css), unsafe_allow_html=True
        )

    def input(
        self,
        label,
        value="",
        key=None,
        on_click=None,
        args=None,
        kwargs=None,
        classes=None,
    ):
        """
        This method will be deprecated in a future version in
        favor of using streamlit widgets.
        """

        return self._root.text_input(
            label, value=value, key=key, on_change=on_click, args=args, kwargs=kwargs
        )

    def checkbox(
        self,
        label,
        value=False,
        key=None,
        on_click=None,
        args=None,
        kwargs=None,
        classes=None,
    ):
        """
        This method will be deprecated in a future version in
        favor of using streamlit widgets.
        """

        return self._root.checkbox(
            label, value=value, key=key, on_change=on_click, args=args, kwargs=kwargs
        )

    def button(
        self, label, key=None, on_click=None, args=None, kwargs=None, classes=None
    ):
        """
        This method will be deprecated in a future version in
        favor of using streamlit widgets.
        """

        return self._root.button(
            label, key=key, on_click=on_click, args=args, kwargs=kwargs
        )

    def reset_progress(self, value=-1):
        """
        This method has beeen deprecated.
        """
        LOGGER.warning("ui.reset_progress() has been deprecated")

    def get_progress(self):
        """
        This method has been deprecated.
        """
        LOGGER.warning("ui.get_progress() has been deprecated")

    def progress(
        self, label, percent, on_load=None, args=None, kwargs=None, classes=None
    ):
        """
        This method has been deprecated.
        """
        LOGGER.warning("ui.progress() has been deprecated")

    def columns(self, items, classes=None, **styles):
        """
        This method will be deprecated in a future version in
        favor of using streamlit widgets.
        """

        return [UI(column) for column in self._root.columns(items)]

    def dropdown(
        self,
        label,
        options,
        index=0,
        format_func=str,
        key=None,
        on_change=None,
        args=None,
        kwargs=None,
        multiple=False,
        classes=None,
    ):
        """
        This method will be deprecated in a future version in
        favor of using streamlit widgets.
        """
        from comet_ml import APIExperiment

        def get_name(experiment):
            return experiment.name

        if (
            len(options) > 0
            and isinstance(options[0], APIExperiment)
            and format_func is str
        ):

            format_func = get_name

        if not multiple:
            return self._root.selectbox(
                label=label,
                options=options,
                index=index,
                format_func=format_func,
                key=key,
                on_change=on_change,
                args=args,
                kwargs=kwargs,
            )
        else:
            default = index if index else options[:1]
            return self._root.multiselect(
                label=label,
                options=options,
                default=default,
                format_func=format_func,
                key=key,
                on_change=on_change,
                args=args,
                kwargs=kwargs,
            )

    def display_plotly_figure(self, figure):
        """
        This method will be deprecated in a future version in
        favor of using streamlit widgets.
        """

        return self._root.plotly_chart(figure, use_container_width=True)

    def display_figure(self, plt):  # matplotlib figure or plt
        """
        This method will be deprecated in a future version in
        favor of using streamlit widgets.
        """

        try:
            fig = plt.gcf()
        except Exception:
            fig = plt

        return self._root.pyplot(fig)

    def display_image(self, data, format=None, width=None):
        """
        This method will be deprecated in a future version in
        favor of using streamlit widgets.
        """
        if format == "svg" and isinstance(data, bytes):
            data = data.decode("utf")

        self._root.image(data)

    def display_text(self, *lines):
        """
        This method will be deprecated in a future version in
        favor of using streamlit widgets.
        """

        self._root.write("\n".join(str(line) for line in lines))

    def display_markdown(self, *args):
        """
        This method will be deprecated in a future version in
        favor of using streamlit widgets.
        """
        for item in args:
            self._root.markdown(item)

    def display(self, *args, format=None, **kwargs):
        """
        This method will be deprecated in a future version in
        favor of using streamlit widgets.
        """
        if format == "markdown":
            self.display_markdown(*args)
        elif all(isinstance(arg, str) for arg in args):
            data = " ".join([str(arg) for arg in args])
            self._display_one(data, format, **kwargs)
        else:
            for data in args:
                self._display_one(data, format, **kwargs)

    def _display_one(self, data, format=None, **kwargs):
        obj_type = _get_object_type(data)
        if hasattr(data, "_repr_png_"):
            self.display_image(data, **kwargs)
        elif obj_type in [
            "matplotlib.figure.Figure",
            "matplotlib.pyplot.module",
        ]:
            self.display_figure(data)
        elif hasattr(data, "__class__") and (data.__class__.__name__ == "Figure"):
            # assume plotly
            self.display_plotly_figure(data)
        elif hasattr(data, "__class__") and (data.__class__.__name__ == "DataFrame"):
            # dataframe, check before to_html, _repr_html_
            self._root.write(data)
        elif hasattr(data, "_repr_html_"):
            self._root.write(data._repr_html_(), unsafe_allow_html=True)
        elif hasattr(data, "to_html"):
            self._root.write(data.to_html(), unsafe_allow_html=True)
        elif hasattr(data, "_repr_json_"):
            self.display_text(str(data._repr_json_()))
        elif hasattr(data, "_repr_jpeg_"):
            self.display_image(data._repr_jpg_(), format="jpg", **kwargs)
        elif hasattr(data, "_repr_svg_"):
            self._root.write(data._repr_svg_())
        elif format == "text":
            self.display_text(data)
        else:
            self._root.write(data, unsafe_allow_html=True)
