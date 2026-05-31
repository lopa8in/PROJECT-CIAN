# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2025 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
from .config_constants import (
    ADDITIONAL_STREAMER_UPLOAD_TIMEOUT,
    ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_INTERVAL_SECONDS,
    ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_MAX_BATCH_SIZE,
    COMET_URL_OVERRIDE_CONFIG_KEY,
    DEFAULT_ARTIFACT_DOWNLOAD_TIMEOUT,
    DEFAULT_FILE_UPLOAD_READ_TIMEOUT,
    DEFAULT_OFFLINE_DATA_DIRECTORY,
    DEFAULT_POOL_RATIO,
    DEFAULT_REMOTE_MODEL_DOWNLOAD_TIMEOUT,
    DEFAULT_STREAMER_BEAT_DURATION,
    DEFAULT_STREAMER_MSG_TIMEOUT,
    DEFAULT_SYSTEM_MONITORING_INTERVAL,
    DEFAULT_WAIT_FOR_FINISH_SLEEP_INTERVAL,
    FALLBACK_STREAMER_CONNECTION_CHECK_INTERVAL_SECONDS,
    GET_OR_ADD_RUN_RETRY_BACKOFF_FACTOR,
    GET_OR_ADD_RUN_RETRY_BACKOFF_MAX,
    GET_OR_ADD_RUN_RETRY_CONNECT,
    GET_OR_ADD_RUN_RETRY_READ,
    GET_OR_ADD_RUN_RETRY_STATUS,
    GET_OR_ADD_RUN_RETRY_TOTAL,
    HTTP_SESSION_RETRY_BACKOFF_FACTOR_DEFAULT,
    HTTP_SESSION_RETRY_TOTAL_DEFAULT,
    MAXIMAL_SVG_FIGURE_SIZE,
    MESSAGE_BATCH_METRIC_INTERVAL_SECONDS,
    MESSAGE_BATCH_METRIC_MAX_BATCH_SIZE,
    MESSAGE_BATCH_PARAMETERS_INTERVAL_SECONDS,
    MESSAGE_BATCH_STDOUT_INTERVAL_SECONDS,
    MESSAGE_BATCH_STDOUT_MAX_BATCH_SIZE,
    MESSAGE_BATCH_USE_COMPRESSION_DEFAULT,
    S3_MULTIPART_EXPIRES_IN,
    S3_MULTIPART_SIZE_THRESHOLD_DEFAULT,
)

# Represents dictionary with all configuration options available
#
CONFIG_MAP = {
    "comet.disable_auto_logging": {"type": int, "default": 0},
    "comet.api_key": {"type": str},
    "comet.offline_directory": {"type": str, "default": DEFAULT_OFFLINE_DATA_DIRECTORY},
    "comet.git_directory": {"type": str},
    "comet.offline_sampling_size": {"type": int, "default": 15000},
    COMET_URL_OVERRIDE_CONFIG_KEY: {
        "type": str,
        "default": None,
    },
    "comet.optimizer_url": {
        "type": str,
        "default": None,
    },
    "comet.experiment_key": {"type": str},
    "comet.project_name": {"type": str},
    "comet.workspace": {"type": str},
    "comet.experiment_name": {"type": str},
    "comet.display_summary_level": {"type": int, "default": 1},
    # Logging
    "comet.logging.file": {"type": str},
    "comet.logging.file_level": {"type": str, "default": "INFO"},
    "comet.logging.file_overwrite": {"type": bool, "default": False},
    "comet.logging.hide_api_key": {"type": bool, "default": True},
    "comet.logging.console": {"type": str},
    "comet.logging.metrics_ignore": {
        "type": list,
        "default": "keras:batch_size,keras:batch_batch",
    },
    "comet.logging.parameters_ignore": {
        "type": list,
        "default": "keras:verbose,keras:do_validation,keras:validation_steps",
    },
    "comet.logging.others_ignore": {"type": list, "default": ""},
    # Timeout, unit is seconds
    "comet.timeout.cleaning": {"type": int, "default": DEFAULT_STREAMER_MSG_TIMEOUT},
    "comet.timeout.upload": {
        "type": int,
        "default": ADDITIONAL_STREAMER_UPLOAD_TIMEOUT,
    },
    "comet.timeout.http": {"type": int, "default": 10},
    "comet.optimizer_timeout.http": {"type": int, "default": 30},
    "comet.timeout.api": {"type": int, "default": 10},
    "comet.timeout.file_upload": {
        "type": float,
        "default": DEFAULT_FILE_UPLOAD_READ_TIMEOUT,
    },
    "comet.timeout.file_download": {"type": int, "default": 600},
    "comet.timeout.artifact_download": {
        "type": int,
        "default": DEFAULT_ARTIFACT_DOWNLOAD_TIMEOUT,
    },
    "comet.timeout.remote_model_download": {
        "type": int,
        "default": DEFAULT_REMOTE_MODEL_DOWNLOAD_TIMEOUT,
    },
    # Sleep intervals
    "comet.internal.streamer_wait_sleep_interval": {
        "type": int,
        "default": DEFAULT_WAIT_FOR_FINISH_SLEEP_INTERVAL,
    },
    "comet.internal.streamer_beat_duration": {
        "type": int,
        "default": DEFAULT_STREAMER_BEAT_DURATION,
    },
    # HTTP Allow header
    "comet.allow_header.name": {"type": str},
    "comet.allow_header.value": {"type": str},
    # Backend minimal rest V2 version
    "comet.rest_v2_minimal_backend_version": {"type": str, "default": "1.2.78"},
    # Feature flags
    "comet.override_feature.sdk_http_logging": {
        "type": bool
    },  # Leave feature toggle default to None
    "comet.override_feature.sdk_announcement": {
        "type": bool
    },  # Leave feature toggle default to None
    # Experiment log controls:
    "comet.system_cpu_interval": {
        "type": int,
        "default": DEFAULT_SYSTEM_MONITORING_INTERVAL,
    },
    "comet.system_gpu_interval": {
        "type": int,
        "default": DEFAULT_SYSTEM_MONITORING_INTERVAL,
    },
    "comet.system_network_interval": {
        "type": int,
        "default": DEFAULT_SYSTEM_MONITORING_INTERVAL,
    },
    "comet.system_disk_interval": {
        "type": int,
        "default": DEFAULT_SYSTEM_MONITORING_INTERVAL,
    },
    "comet.start.online": {"type": bool},
    "comet.start.mode": {"type": str},
    "comet.start.experiment_name": {"type": str},
    "comet.start.experiment_tags": {"type": list},
    "comet.auto_log.cli_arguments": {"type": bool},
    "comet.auto_log.code": {"type": bool},
    "comet.auto_log.disable": {"type": bool},
    "comet.auto_log.env_cpu": {"type": bool},
    "comet.auto_log.env_cpu_per_core": {"type": bool, "default": False},
    "comet.auto_log.env_details": {"type": bool},
    "comet.auto_log.env_gpu": {"type": bool},
    "comet.auto_log.env_host": {"type": bool},
    "comet.auto_log.env_network": {"type": bool},
    "comet.auto_log.env_disk": {"type": bool},
    "comet.auto_log.git_metadata": {"type": bool},
    "comet.auto_log.git_patch": {"type": bool},
    "comet.auto_log.graph": {"type": bool},
    "comet.auto_log.metrics": {"type": bool},
    "comet.auto_log.figures": {"type": bool, "default": True},
    "comet.auto_log.output_logger": {"type": str},
    "comet.auto_log.parameters": {"type": bool},
    "comet.auto_log.histogram_tensorboard": {"type": bool, "default": False},
    "comet.auto_log.histogram_epoch_rate": {"type": int, "default": 1},
    "comet.auto_log.histogram_weights": {"type": bool, "default": False},
    "comet.auto_log.histogram_gradients": {"type": bool, "default": False},
    "comet.auto_log.histogram_activations": {"type": bool, "default": False},
    "comet.keras.histogram_name_prefix": {
        "type": str,
        "default": "{layer_num:0{max_digits}d}",
    },
    "comet.keras.histogram_activation_index_list": {"type": "int_list", "default": "0"},
    "comet.keras.histogram_activation_layer_list": {"type": list, "default": "-1"},
    "comet.keras.histogram_batch_size": {"type": int, "default": 1000},
    "comet.keras.histogram_gradient_index_list": {"type": "int_list", "default": "0"},
    "comet.keras.histogram_gradient_layer_list": {"type": list, "default": "-1"},
    "comet.auto_log.metric_step_rate": {"type": int, "default": 10},
    "comet.auto_log.co2": {"type": bool},
    "comet.auto_log.tfma": {"type": bool, "default": False},
    "comet.distributed_node_identifier": {"type": str},
    # Internals:
    "comet.internal.reporting": {"type": bool, "default": True},
    "comet.internal.file_upload_worker_ratio": {
        "type": int,
        "default": DEFAULT_POOL_RATIO,
    },
    "comet.internal.worker_count": {"type": int},
    "comet.internal.check_tls_certificate": {"type": bool, "default": True},
    "comet.internal.sentry_dsn": {
        "type": str,
        "default": "https://55e8a7aaa6bfdbaead68218e43b8615c@o168229.ingest.us.sentry.io/4505505645330432",
    },
    "comet.internal.sentry_debug": {"type": bool, "default": False},
    "comet.internal.max_svg_figure_size": {
        "type": int,
        "default": MAXIMAL_SVG_FIGURE_SIZE,
    },
    "comet.include.custom_ca_bundle": {"type": str, "default": None},
    # Deprecated:
    "comet.display_summary": {"type": bool, "default": None},
    "comet.auto_log.weights": {"type": bool, "default": None},
    # Related to `comet_ml.start`
    "comet.resume_strategy": {"type": str, "default": None},
    "comet.offline": {"type": bool, "default": False},
    # Error tracking.
    "comet.error_tracking.enable": {"type": bool},
    # Related to message batch processing
    "comet.message_batch.use_compression": {
        "type": bool,
        "default": MESSAGE_BATCH_USE_COMPRESSION_DEFAULT,
    },
    "comet.message_batch.metric_interval": {
        "type": float,
        "default": MESSAGE_BATCH_METRIC_INTERVAL_SECONDS,
    },
    "comet.message_batch.metric_max_size": {
        "type": int,
        "default": MESSAGE_BATCH_METRIC_MAX_BATCH_SIZE,
    },
    "comet.message_batch.parameters_interval": {
        "type": int,
        "default": MESSAGE_BATCH_PARAMETERS_INTERVAL_SECONDS,
    },
    "comet.message_batch.stdout_interval": {
        "type": int,
        "default": MESSAGE_BATCH_STDOUT_INTERVAL_SECONDS,
    },
    "comet.message_batch.stdout_max_size": {
        "type": int,
        "default": MESSAGE_BATCH_STDOUT_MAX_BATCH_SIZE,
    },
    "comet.message_batch.artifact_remote_assets_interval": {
        "type": int,
        "default": ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_INTERVAL_SECONDS,
    },
    "comet.message_batch.artifact_remote_assets_max_size": {
        "type": int,
        "default": ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_MAX_BATCH_SIZE,
    },
    # Fallback streamer
    "comet.fallback_streamer.connection_check_interval": {
        "type": int,
        "default": FALLBACK_STREAMER_CONNECTION_CHECK_INTERVAL_SECONDS,
    },
    "comet.fallback_streamer.keep_offline_zip": {
        "type": bool,
        "default": False,
    },
    "comet.fallback_streamer.fallback_to_offline_min_backend_version": {
        "type": str,
        "default": "3.3.11",
    },
    "comet.novel_model_registry_api.minimum_backend_version": {
        "type": str,
        "default": "3.5.42",
    },
    "comet.api_experiment.delete_tags_minimum_backend_version": {
        "type": str,
        "default": "3.12.26",
    },
    "comet.api_experiment.delete_parameters_minimum_backend_version": {
        "type": str,
        "default": "3.29.724",
    },
    "comet.api_experiment.get_all_experiment_metrics_minimum_backend_version": {
        "type": str,
        "default": "3.20.60",
    },
    "comet.artifact.remote_assets_batch_minimum_backend_version": {
        "type": str,
        "default": "3.12.26",
    },
    "comet.internal.raw_throttling_messages_minimum_backend_version": {
        "type": str,
        "default": "3.62.294",
    },
    "comet.disable_announcement": {"type": bool, "default": False},
    "comet.rich_output": {"type": bool, "default": True},
    "comet.s3_multipart.size_threshold": {
        "type": int,
        "default": S3_MULTIPART_SIZE_THRESHOLD_DEFAULT,
    },
    "comet.s3_multipart.expires_in": {
        "type": int,
        "default": S3_MULTIPART_EXPIRES_IN,
    },
    "comet.s3_direct_multipart.upload_enabled": {
        "type": bool,
        "default": False,
    },
    "comet.http_session.retry_total": {
        "type": int,
        "default": HTTP_SESSION_RETRY_TOTAL_DEFAULT,
    },
    "comet.http_session.retry_backoff_factor": {
        "type": int,
        "default": HTTP_SESSION_RETRY_BACKOFF_FACTOR_DEFAULT,
    },
    "comet.get_or_add_experiment.retry_connect": {
        "type": int,
        "default": GET_OR_ADD_RUN_RETRY_CONNECT,
    },
    "comet.get_or_add_experiment.retry_read": {
        "type": int,
        "default": GET_OR_ADD_RUN_RETRY_READ,
    },
    "comet.get_or_add_experiment.retry_status": {
        "type": int,
        "default": GET_OR_ADD_RUN_RETRY_STATUS,
    },
    "comet.get_or_add_experiment.retry_total": {
        "type": int,
        "default": GET_OR_ADD_RUN_RETRY_TOTAL,
    },
    "comet.get_or_add_experiment.retry_backoff_factor": {
        "type": int,
        "default": GET_OR_ADD_RUN_RETRY_BACKOFF_FACTOR,
    },
    "comet.get_or_add_experiment.retry_backoff_max": {
        "type": int,
        "default": GET_OR_ADD_RUN_RETRY_BACKOFF_MAX,
    },
}
