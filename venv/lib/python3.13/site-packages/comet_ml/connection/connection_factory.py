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
from typing import Any, Dict, Optional

from ..config import (
    get_check_tls_certificate,
    get_comet_url_override,
    get_config,
    get_optimizer_address,
)
from ..config.config_api import get_api_key, get_backend_address, get_comet_timeout_http
from ..logging_messages import CONNECTION_GET_REST_API_CLIENT_NO_API_KEY_EXCEPTION
from ..utils import get_root_url
from .connection_api import (
    CometApiClient,
    LowLevelHTTPClient,
    OptimizerAPI,
    RestApiClient,
    RestApiClientWithCache,
    RestServerConnection,
)


def get_optimizer_api(
    api_key: str,
    server_address: Optional[str] = None,
    headers: Optional[Dict[str, Any]] = None,
) -> OptimizerAPI:
    config = get_config()
    if server_address is None:
        server_address = get_optimizer_address(config)

    server_url = get_root_url(server_address)

    low_level_api = LowLevelHTTPClient(
        api_key=api_key,
        server_address=server_url,
        default_timeout=config["comet.optimizer_timeout.http"],
        headers=headers,
        verify_tls=get_check_tls_certificate(config),
        config=config,
    )

    return OptimizerAPI(api_key, low_level_api, server_address)


def get_comet_api_client(
    server_address: Optional[str] = None, headers: Optional[Dict[str, Any]] = None
) -> CometApiClient:
    """Factory to create `CometApiClient` instance which provides methods to communicate with comet API
    that have `api/auth/` path, for example: https://comet.com/api/auth/login
    """
    config = get_config()
    if server_address is None:
        server_address = get_comet_url_override(config)
    server_url = get_root_url(server_address)

    low_level_api = LowLevelHTTPClient(
        api_key=get_api_key(None, config),
        server_address=server_url,
        default_timeout=config["comet.timeout.http"],
        headers=headers,
        verify_tls=config.get_bool(None, "comet.internal.check_tls_certificate"),
        config=config,
    )

    return CometApiClient(server_url, low_level_api, config)


def get_rest_api_client(
    version: str,
    server_address: Optional[str] = None,
    headers: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    use_cache: bool = False,
    check_version: bool = True,
    retry_auth_errors: bool = False,
) -> RestApiClient:
    """Factory method to create `RestApiClient` instance which provides methods for REST API data flows.
    The path of REST API endpoints has `rest/v2` part, e.g:
    `https://comet.com/clientlib/rest/v2/write/experiment/output` (recent code blocks)
    `https://comet.com/api/rest/v2/experiment/metadata` (old code blocks)
    """
    settings = get_config()
    if api_key is None:
        raise ValueError(CONNECTION_GET_REST_API_CLIENT_NO_API_KEY_EXCEPTION)

    if server_address is None:
        server_address = get_comet_url_override(settings)
    server_url = get_root_url(server_address)

    low_level_api = LowLevelHTTPClient(
        api_key=api_key,
        server_address=server_url,
        default_timeout=settings["comet.timeout.api"],
        headers=headers,
        verify_tls=get_check_tls_certificate(settings),
        retry_auth_errors=retry_auth_errors,
        config=settings,
    )

    if use_cache:
        return RestApiClientWithCache(
            server_url=server_url,
            version=version,
            low_level_api_client=low_level_api,
            api_key=api_key,
            config=settings,
            check_version=check_version,
        )
    else:
        return RestApiClient(
            server_url=server_url,
            version=version,
            low_level_api_client=low_level_api,
            api_key=api_key,
            config=settings,
            check_version=check_version,
        )


def get_rest_server_connection(
    api_key: str,
    experiment_key: Optional[str],
    server_address: Optional[str] = None,
    default_timeout: Optional[float] = None,
    verify_tls: Optional[bool] = None,
) -> RestServerConnection:
    """Factory method to create `RestServerConnection` instance which provides methods for REST API data flows for
    endpoints that has `logger/` part, e.g: `https://www.comet.com/clientlib/logger/add/run`
    """

    settings = get_config()

    if server_address is None:
        server_address = get_backend_address(settings)

    if default_timeout is None:
        default_timeout = get_comet_timeout_http(settings)

    if verify_tls is None:
        verify_tls = get_check_tls_certificate(settings)

    return RestServerConnection(
        api_key=api_key,
        experiment_id=experiment_key,
        server_address=server_address,
        default_timeout=default_timeout,
        verify_tls=verify_tls,
        config=settings,
    )
