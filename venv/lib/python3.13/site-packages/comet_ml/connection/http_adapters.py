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
from typing import Optional

from requests.adapters import HTTPAdapter
from requests_toolbelt.adapters.socket_options import TCPKeepAliveAdapter


class HTTPSAdapter(HTTPAdapter):

    def __init__(self, custom_ca_certs: Optional[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_ca_certs = custom_ca_certs

    def cert_verify(self, conn, url, verify, cert):
        # Tests for existence due to issue when Session changes
        # This can happen in Comet's Python Panels
        if getattr(self, "_custom_ca_certs", None) is None:
            super().cert_verify(conn, url, verify, cert)
            return

        # use our custom CA certificates path
        super().cert_verify(
            conn=conn,
            url=url,
            verify=self._custom_ca_certs,
            cert=cert,
        )


class HTTPSTCPKeepAliveAdapter(TCPKeepAliveAdapter):

    def __init__(self, custom_ca_certs: Optional[str], **kwargs):
        super().__init__(**kwargs)
        self._custom_ca_certs = custom_ca_certs

    def cert_verify(self, conn, url, verify, cert):
        if self._custom_ca_certs is None:
            super().cert_verify(conn, url, verify, cert)
            return

        # use our custom CA certificates path
        super().cert_verify(
            conn=conn,
            url=url,
            verify=self._custom_ca_certs,
            cert=cert,
        )
