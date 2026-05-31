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
import logging
import platform
import threading
import warnings
from typing import List, Optional

from comet_ml import authhook

import urllib3
from requests import Session
from requests.adapters import HTTPAdapter
from requests_toolbelt.adapters.socket_options import TCPKeepAliveAdapter
from urllib3 import Retry

from .. import semantic_version
from ..authhook import aws_sagemaker
from ..config import Config, get_config
from ..utils import get_comet_version
from . import retry_with_full_jitter
from .ca_certs import setup_ca_certs
from .http_adapters import HTTPSAdapter, HTTPSTCPKeepAliveAdapter

STATUS_FORCELIST_NO_AUTH_ERRORS = [500, 502, 503, 504]
STATUS_FORCELIST_FULL = [401, 403]
STATUS_FORCELIST_FULL.extend(STATUS_FORCELIST_NO_AUTH_ERRORS)

API_KEY_HEADER = "Authorization"

LOGGER = logging.getLogger(__name__)


def setup_http_session_authentication(session: Session) -> None:
    """
    Sets up HTTP session authentication for accessing APIs.

    Args:
        session (Session): The HTTP session object that requires authentication setup.

    This function ensures authentication for accessing AWS SageMaker and integrates authentication hooks into the session.
    It checks if the environment is within AWS SageMaker and performs login operations specific to AWS SageMaker if True.

    Steps:
        1. Checks and performs late login to AWS SageMaker to read environment configuration values that might be set post-import.
        2. Integrates authentication hooks into the provided HTTP session.
    """
    if aws_sagemaker.is_in_aws_sagemaker():
        aws_sagemaker.login_aws_sagemaker()

    authhook.http_session_hook(session=session)


def get_comet_http_session(
    api_key: Optional[str],
    retry_strategy: Optional[Retry] = None,
    verify_tls: bool = True,
    tcp_keep_alive: bool = False,
    config: Optional[Config] = None,
) -> Session:
    """Creates http session with Comet related headers set and authentication hook enabled."""
    if config is None:
        config = get_config()  # This can be slow if called for every new session

    session = get_http_session(
        retry_strategy=retry_strategy,
        verify_tls=verify_tls,
        tcp_keep_alive=tcp_keep_alive,
        config=config,
    )

    # Add default debug headers
    session.headers.update(
        {
            "X-COMET-DEBUG-SDK-VERSION": get_comet_version(),
            "X-COMET-DEBUG-PY-VERSION": platform.python_version(),
        }
    )

    # Add authorization header
    if api_key is not None:
        session.headers.update({API_KEY_HEADER: api_key})

    # Setup HTTP allow header if configured
    allow_header_name = config["comet.allow_header.name"]
    allow_header_value = config["comet.allow_header.value"]

    if allow_header_name and allow_header_value:
        session.headers[allow_header_name] = allow_header_value

    setup_http_session_authentication(session)

    return session


def get_http_session(
    config: Config,
    retry_strategy: Optional[Retry] = None,
    verify_tls: bool = True,
    tcp_keep_alive: bool = False,
) -> Session:
    """Returns http session with custom HTTPS adapter and optional retry strategy defined"""
    session = Session()

    custom_ca_certs = config["comet.include.custom_ca_bundle"]
    if custom_ca_certs is not None:
        custom_ca_certs = setup_ca_certs(custom_ca_certs)

    # Setup retry strategy if asked
    if tcp_keep_alive is True:
        http_adapter = TCPKeepAliveAdapter(
            idle=60, count=5, interval=60, max_retries=retry_strategy
        )
        https_adapter = HTTPSTCPKeepAliveAdapter(
            custom_ca_certs=custom_ca_certs,
            idle=60,
            count=5,
            interval=60,
            max_retries=retry_strategy,
        )
    elif retry_strategy is not None:
        http_adapter = HTTPAdapter(max_retries=retry_strategy)
        https_adapter = HTTPSAdapter(
            custom_ca_certs=custom_ca_certs, max_retries=retry_strategy
        )
    else:
        http_adapter = HTTPAdapter()
        https_adapter = HTTPSAdapter(custom_ca_certs=custom_ca_certs)

    if http_adapter is not None:
        session.mount("http://", http_adapter)

    if https_adapter is not None:
        session.mount("https://", https_adapter)

    if verify_tls is False:
        # Only set the verify if it's disabled. The current default for the verify attribute is
        # True but this way we will survive any change of the default value
        session.verify = False
        # Also filter the warning that urllib3 emits to not overflow the output with them
        warnings.filterwarnings(
            "ignore", category=urllib3.exceptions.InsecureRequestWarning
        )

    return session


def get_retry_strategy(
    status_forcelist: List[int],
    total_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None,
    config: Optional[Config] = None,
) -> Retry:
    # The total backoff sleeping time is computed like that:
    # backoff = 2
    # retries = 3
    # s = lambda b, i: b * (2 ** (i - 1))
    # sleep = sum(s(backoff, i) for i in range(1, retries + 1))
    # Will wait up to 14s

    if status_forcelist is None:
        status_forcelist = STATUS_FORCELIST_NO_AUTH_ERRORS

    if config is None:
        config = get_config()

    if total_retries is None:
        total_retries = config.get_int(None, "comet.http_session.retry_total")

    if backoff_factor is None:
        backoff_factor = float(
            config.get_int(None, "comet.http_session.retry_backoff_factor")
        )

    if urllib3.__version__.startswith("2."):
        kwargs = {"allowed_methods": None}
    else:
        kwargs = {"method_whitelist": False}

    return Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False,
        **kwargs
    )


def get_retry_strategy_for_get_or_add_run(
    status_forcelist: List[int],
    config: Optional[Config],
) -> Retry:
    # The total backoff sleeping time is computed like that:
    # backoff = 2
    # retries = 3
    # s = lambda b, i: b * (2 ** (i - 1)) * (1 + random.random())
    # sleep = sum(s(backoff, i) for i in range(1, retries + 1))
    # Will wait up to 14s

    connect_retries = config.get_int(None, "comet.get_or_add_experiment.retry_connect")
    status_retries = config.get_int(None, "comet.get_or_add_experiment.retry_status")
    read_retries = config.get_int(None, "comet.get_or_add_experiment.retry_read")
    total_retries = config.get_int(None, "comet.get_or_add_experiment.retry_total")

    backoff_factor = config.get_int(
        None, "comet.get_or_add_experiment.retry_backoff_factor"
    )
    backoff_max = config.get_int(None, "comet.get_or_add_experiment.retry_backoff_max")

    if semantic_version.SemanticVersion.parse(urllib3.__version__) < "1.26.8":
        # urllib3 is already restricted in setup.py to be >=1.26.8, however, users
        # often ignore dependency conflicts and use older version of urllib3
        #
        # This is a fallback mechanism allowing experiments to run but with a
        # different retry strategy (no jittering, which was added in comet-ml==3.49.12).
        LOGGER.warning(
            "urllib3 version is older than 1.26.8, using fallback retry strategy for experiment creation. We recommend upgrading to urllib3>=1.26.8 to have the most stable experience."
        )
        return get_retry_strategy(status_forcelist=status_forcelist, config=config)

    return retry_with_full_jitter.RetryWithFullJitter(
        connect=connect_retries,
        read=read_retries,
        total=total_retries,
        status=status_retries,
        backoff_factor=backoff_factor,
        backoff_max=backoff_max,
        status_forcelist=status_forcelist,
        raise_on_status=False,
        allowed_methods=None,
    )


THREAD_SESSIONS = threading.local()


def get_cached_comet_http_session(
    retry: bool,
    verify_tls: bool,
    api_key: Optional[str],
    tcp_keep_alive: bool = True,
    config: Optional[Config] = None,
) -> Session:
    """Returns reusable Comet specific http session which is cached in thread local dictionary."""
    # As long as the session is not part of a reference loop, the thread local dict will be cleaned
    # up when each thread ends, garbage-collecting the Session object and closing the
    # resources
    session_key = str((retry, tcp_keep_alive, verify_tls, "comet_session"))

    cached_session: Optional[Session] = THREAD_SESSIONS.__dict__.get(session_key, None)

    if cached_session is not None:
        return cached_session

    if config is None:
        config = get_config()

    retry_strategy = False
    if retry is True:
        retry_strategy = get_retry_strategy(
            status_forcelist=STATUS_FORCELIST_NO_AUTH_ERRORS, config=config
        )

    new_session = get_comet_http_session(
        retry_strategy=retry_strategy,
        tcp_keep_alive=tcp_keep_alive,
        verify_tls=verify_tls,
        api_key=api_key,
        config=config,
    )
    THREAD_SESSIONS.__dict__[session_key] = new_session

    return new_session


def get_cached_http_session(
    retry: bool,
    verify_tls: bool,
    tcp_keep_alive: bool = True,
    config: Optional[Config] = None,
) -> Session:
    """Returns reusable plain http session which is cached in thread local dictionary."""
    session_key = str((retry, tcp_keep_alive, verify_tls))

    cached_session: Optional[Session] = THREAD_SESSIONS.__dict__.get(session_key, None)

    if cached_session is not None:
        return cached_session

    if config is None:
        config = get_config()

    retry_strategy = False
    if retry is True:
        retry_strategy = get_retry_strategy(
            status_forcelist=STATUS_FORCELIST_NO_AUTH_ERRORS, config=config
        )

    new_session = get_http_session(
        retry_strategy=retry_strategy,
        tcp_keep_alive=tcp_keep_alive,
        verify_tls=verify_tls,
        config=config,
    )
    THREAD_SESSIONS.__dict__[session_key] = new_session

    return new_session
