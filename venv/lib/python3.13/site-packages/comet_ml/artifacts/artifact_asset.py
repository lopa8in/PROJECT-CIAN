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


class ArtifactAsset:
    """ArtifactAsset(remote, logical_path, size, link, metadata, asset_type, local_path_or_data):
    represent local and remote assets added to an Artifact object but not yet uploaded
    """

    __slots__ = (
        "_asset_id",
        "_remote",
        "_logical_path",
        "_size",
        "_link",
        "_metadata",
        "_asset_type",
        "_local_path_or_data",
    )

    def __init__(
        self,
        remote: bool,
        logical_path: str,
        size: int,
        link: Optional[str],
        metadata: Optional[Dict[str, Any]],
        asset_type: Optional[str],
        local_path_or_data: Optional[Any],
        asset_id: Optional[str] = None,
    ) -> None:
        self._remote = remote
        self._logical_path = logical_path
        self._size = size
        self._link = link
        self._metadata = metadata
        self._asset_type = asset_type
        self._local_path_or_data = local_path_or_data
        self._asset_id = asset_id

    @property
    def id(self) -> Optional[str]:
        return self._asset_id

    @property
    def remote(self) -> bool:
        """Is the asset a remote asset or not, boolean"""
        return self._remote

    @property
    def logical_path(self) -> str:
        """Asset relative logical_path, str or None"""
        return self._logical_path

    @property
    def size(self) -> int:
        """Asset size if the asset is a non-remote asset, int"""
        return self._size

    @property
    def link(self) -> str:
        """Asset remote link if the asset is remote, str or None"""
        return self._link

    @property
    def metadata(self) -> Dict[str, Any]:
        """Asset metadata, dict"""
        return self._metadata

    @property
    def asset_type(self) -> str:
        """Asset type, str"""
        return self._asset_type

    @property
    def local_path_or_data(self) -> Any:
        """Asset local path or in-memory file if the asset is non-remote, str, memory-file or None"""
        return self._local_path_or_data

    def __repr__(self) -> str:
        return (
            "%s(remote=%r, logical_path=%r, size=%r, link=%r, metadata=%r, asset_type=%r, local_path_or_data=%r, asset_id=%r)"
            % (
                self.__class__.__name__,
                self._remote,
                self._logical_path,
                self._size,
                self._link,
                self._metadata,
                self._asset_type,
                self._local_path_or_data,
                self._asset_id,
            )
        )

    def __eq__(self, other: "ArtifactAsset") -> bool:
        return (
            self._remote == other._remote
            and self._logical_path == other._logical_path
            and self._size == other._size
            and self._link == other._link
            and self._metadata == other._metadata
            and self._asset_type == other._asset_type
            and self._local_path_or_data == other._local_path_or_data
            and self._asset_id == other._asset_id
        )

    def __lt__(self, other: "ArtifactAsset") -> bool:
        return self._logical_path < other._logical_path
