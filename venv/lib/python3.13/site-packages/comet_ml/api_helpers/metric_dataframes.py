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
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import pandas as pd

LOGGER = logging.getLogger(__name__)


def get_dataframe_from_multi_metrics(
    multi_metrics: Dict, x_axis: str, columns: List
) -> "pd.DataFrame":
    import pandas as pd

    data = []
    keys = ["experiment_key", x_axis]

    for experiment in multi_metrics.values():
        experiment_key = experiment.get("experimentKey", "")
        experiment_name = experiment.get("experimentName", "")
        experiment_metrics = experiment.get("metrics", [])

        for metric in experiment_metrics:
            metric_name = metric.get("metricName", "")

            if metric_name not in columns:
                continue

            metric_values = metric.get("values", [])
            metric_x_axis = metric.get(x_axis + "s", [])

            for idx, x_axis_value in enumerate(metric_x_axis):
                data.append(
                    {
                        "experiment_key": experiment_key,
                        "experiment_name": experiment_name,
                        metric_name: _parse_value(metric_values[idx]),
                        x_axis: x_axis_value,
                    }
                )

    df = pd.DataFrame(data, columns=columns)

    if df.empty:
        return df[columns]

    df = df[columns].groupby(keys).aggregate("last").reset_index()

    return df[columns]


def interpolate_metric_dataframe(
    df: "pd.DataFrame", x_axis: str, metrics: List[str], columns: List[str]
) -> "pd.DataFrame":
    x_axis_values = _get_x_axis_for_interpolation(df, x_axis)

    interpolated_df = (
        df.groupby(["experiment_key"])
        .apply(
            lambda x: _interpolate(
                df=x, metrics=metrics, x_axis=x_axis, all_x_axis_values=x_axis_values
            )
        )
        .reset_index()
    )

    return interpolated_df.sort_values(["experiment_key", x_axis])[columns]


def metrics_to_total_fidelity_dataframe(
    metrics: List[Dict[str, Any]]
) -> "pd.DataFrame":
    import pandas as pd

    data = []
    for metric in metrics:
        data.append(
            {
                "value": float(metric["metricValue"]),
                "timestamp": metric["timestamp"] / 1000.0,
                "step": metric["step"],
                "epoch": metric["epoch"],
            }
        )
    return pd.DataFrame(data, columns=["value", "timestamp", "step", "epoch"])


def _parse_value(element: Any) -> Any:
    try:
        if element is None:
            return None
        return float(element)
    except ValueError:
        return element


def _get_x_axis_for_interpolation(df: "pd.DataFrame", x_axis: str) -> List:
    filtered_x_axis_df = df.dropna(subset=[x_axis])

    if x_axis == "duration":
        import numpy as np

        return list(
            np.linspace(
                filtered_x_axis_df[x_axis].min(),
                filtered_x_axis_df[x_axis].max(),
                num=100,
            )
        )

    return list(filtered_x_axis_df[x_axis].unique())


def _interpolate(
    df: "pd.DataFrame", metrics: List[str], x_axis: str, all_x_axis_values: List
) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    interpolated_dataframe = pd.DataFrame({x_axis: all_x_axis_values})

    for metric in metrics:
        if pd.api.types.is_numeric_dtype(df[metric]):
            filtered_metric_value_df = df[df[metric].notnull()]

            interpolated_dataframe[metric] = np.interp(
                all_x_axis_values,
                filtered_metric_value_df[x_axis],
                filtered_metric_value_df[metric],
                right=np.nan,
                left=np.nan,
            )
        else:
            interpolated_dataframe[metric] = (
                df.set_index(x_axis).reindex(all_x_axis_values).reset_index()[metric]
            )

    interpolated_dataframe["experiment_name"] = df.iloc[0]["experiment_name"]

    return interpolated_dataframe
