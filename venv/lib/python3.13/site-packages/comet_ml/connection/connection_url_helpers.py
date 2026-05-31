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
from ..constants import ASSET_TYPE_ASSET_ITEM, ASSET_TYPE_VISUALIZATION
from ..utils import url_join

SYSTEM_DETAILS_ENDPOINT = "experiment/system-details"
SYSTEM_DETAILS_WRITE_ENDPOINT = "write/" + SYSTEM_DETAILS_ENDPOINT


def notify_url(server_address):
    return url_join(server_address, "notify/event")


def log_url(server_address: str) -> str:
    return url_join(server_address, "log/add")


def offline_experiment_times_url(server_address: str) -> str:
    return url_join(server_address, "status-report/offline-metadata")


def pending_rpcs_url(server_address: str) -> str:
    return url_join(server_address, "rpc/get-pending-rpcs")


def add_tags_url(server_address: str) -> str:
    return url_join(server_address, "tags/add-tags-to-experiment")


def get_tags_url(server_address: str) -> str:
    return url_join(server_address, "experiment/tags")


def register_rpc_url(server_address: str) -> str:
    return url_join(server_address, "rpc/register-rpc")


def rpc_result_url(server_address: str) -> str:
    return url_join(server_address, "rpc/save-rpc-result")


def new_symlink_url(server_address: str) -> str:
    return url_join(server_address, "symlink/new")


def add_run_url(server_address: str) -> str:
    return url_join(server_address, "logger/add/run")


def get_points_3d_upload_limits_url(server_address: str) -> str:
    return url_join(server_address, "asset/3d-limits")


def get_ping_backend_url(server_address: str) -> str:
    return url_join(server_address, "health/ping")


def get_backend_version_url(server_address: str) -> str:
    return url_join(server_address, "clientlib/isAlive/ver")


def get_run_url(server_address: str) -> str:
    return url_join(server_address, "logger/get/run")


def copy_experiment_url(server_address: str) -> str:
    return url_join(server_address, "logger/copy-steps-from-experiment")


def notification_url(server_address: str) -> str:
    return server_address + "notification/experiment"


def metrics_batch_url(server_address: str) -> str:
    return url_join(server_address, "batch/logger/experiment/metric")


def parameters_batch_url(server_address: str) -> str:
    return url_join(server_address, "batch/logger/experiment/parameter")


def create_asset_url(server_address: str) -> str:
    return url_join(server_address, "asset")


def upload_thumbnail_url(server_address: str, asset_id: str) -> str:
    return url_join(server_address, "asset/%s/thumbnail" % asset_id)


def status_report_update_url(server_address: str) -> str:
    return url_join(server_address, "status-report/update")


def visualization_upload_url():
    """Return the URL to upload visualizations"""
    return "visualizations/upload"


def asset_item_upload_url():
    """Return the URL to upload asset-item"""
    return "asset/asset-item"


def asset_upload_url():
    return "asset/upload"


def get_git_patch_upload_endpoint():
    return "git-patch/upload"


UPLOAD_TYPE_URL_MAP = {
    "git-patch": get_git_patch_upload_endpoint(),
    "shap": visualization_upload_url(),
    "prophet": visualization_upload_url(),
    ASSET_TYPE_VISUALIZATION: visualization_upload_url(),
    ASSET_TYPE_ASSET_ITEM: asset_item_upload_url(),
}


def get_upload_url(server_address: str, upload_type: str) -> str:
    # Default upload url is asset_upload_url
    return url_join(
        server_address, UPLOAD_TYPE_URL_MAP.get(upload_type, asset_upload_url())
    )
