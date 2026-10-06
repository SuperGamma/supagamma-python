"""Stream a response body to a file without holding it in memory.

Why this exists: ``client.download.*`` used to read ``response.content`` in full,
so a pull of a few hundred megabytes needed a few hundred megabytes of RAM (twice,
briefly, while parsing), and ``DownloadResult`` could not be avoided. The streaming
path writes each chunk as it arrives.

Three properties matter more than the chunking, because the bytes being written
were already paid for:

* **All or nothing.** The body goes to a hidden temp file in the destination
  directory and is moved into place with :func:`os.replace` only once it is
  complete. A failure part-way (a reset connection, a full disk, Ctrl-C) removes
  the temp file and leaves the destination exactly as it was, so an existing file
  is never replaced by a truncated one and nothing truncated is ever named like
  the real thing.
* **A short body is an error.** When the server declared a ``Content-Length`` and
  fewer bytes arrived, that raises instead of returning a file that merely looks
  complete.
* **The server never chooses the directory.** A ``Content-Disposition`` filename
  is reduced to its last path component before it is used.

None of this retries. A failure here is ambiguous about money (the debit happens
before the body finishes arriving), and the only safe recovery is to replay the
byte-identical request, which the caller decides.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import httpx

from ._errors import APIConnectionError, APITimeoutError

__all__ = [
    "CHUNK_SIZE",
    "DEFAULT_FILENAME",
    "DownloadedFile",
    "PathLike",
    "awrite_stream",
    "filename_from",
    "resolve_target",
    "safe_filename",
    "write_stream",
]

#: Bytes read from the socket per iteration. Large enough that the per-chunk
#: overhead is noise, small enough that memory stays flat.
CHUNK_SIZE = 1 << 20

DEFAULT_FILENAME = "supagamma-download"

PathLike = Union[str, "os.PathLike[str]"]


@dataclass
class DownloadedFile:
    """A response that was streamed to disk. Nothing is held in memory.

    The result of any download method called with ``save_to=``. It is
    :class:`os.PathLike`, so it can be passed straight to ``open()``,
    ``pandas.read_parquet`` and the like.
    """

    path: Path
    size: int
    content_type: str = ""
    filename: Optional[str] = None
    request_id: Optional[str] = None

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    def __len__(self) -> int:
        return self.size


def filename_from(response: httpx.Response) -> Optional[str]:
    """The ``filename`` in the response's ``Content-Disposition``, unmodified."""
    disposition = response.headers.get("content-disposition", "")
    if "filename=" not in disposition:
        return None
    return disposition.split("filename=", 1)[1].strip().strip('"') or None


def safe_filename(name: Optional[str]) -> str:
    """A bare file name from a server-supplied one: never a path.

    ``../../x`` and ``C:\\x`` both reduce to ``x``; a name that reduces to nothing
    falls back to :data:`DEFAULT_FILENAME`.
    """
    if not name:
        return DEFAULT_FILENAME
    cleaned = Path(name.replace("\\", "/")).name
    return cleaned if cleaned not in ("", ".", "..") else DEFAULT_FILENAME


def resolve_target(path: PathLike, filename: Optional[str]) -> Path:
    """Where to write: ``path`` itself, or the server's name inside it if it is a directory."""
    raw = os.fspath(path)
    target = Path(raw)
    if target.is_dir() or raw.endswith(("/", os.sep)):
        target = target / safe_filename(filename)
    return target


def _partial_for(target: Path) -> Path:
    return target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.part")


def _check_complete(response: httpx.Response, written: int, request_id: Optional[str]) -> None:
    """Fail a body that ended before the length the server declared.

    ``num_bytes_downloaded`` counts raw bytes off the wire, so it is comparable to
    ``Content-Length`` even when the body was content-encoded. It stays at zero for a
    response that arrived already read (a caching transport, a test transport, any
    ``http_client`` the caller injected), so for those the bytes written stand in
    for it, unless the body was decoded on the way and the two cannot be compared.
    """
    declared = response.headers.get("content-length")
    if declared is None or not declared.isdigit():
        return
    received = response.num_bytes_downloaded
    if received == 0 and written > 0:
        if response.headers.get("content-encoding", "identity").strip().lower() not in (
            "",
            "identity",
        ):
            return
        received = written
    if received != int(declared):
        raise APIConnectionError(
            f"incomplete download: received {received} of {declared} bytes",
            request_id=request_id,
        )


def transport_failure(exc: httpx.HTTPError, request_id: Optional[str]) -> APIConnectionError:
    """Map an httpx transport error to the SDK's own, so callers catch one hierarchy.

    The client maps the errors it sees up to the response headers. Once a body is
    streaming, a drop or a stall surfaces from the iteration instead, so the
    writers below map those with this too. So does a fetch the client does not
    make itself (the signed URL of an export).
    """
    if isinstance(exc, httpx.TimeoutException):
        return APITimeoutError(str(exc), request_id=request_id)
    return APIConnectionError(str(exc), request_id=request_id)


def write_stream(
    response: httpx.Response, path: PathLike, *, filename: Optional[str] = None
) -> DownloadedFile:
    """Write an open streamed response to ``path`` and return what was written."""
    request_id = response.headers.get("x-request-id")
    target = resolve_target(path, filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_for(target)
    written = 0
    try:
        try:
            with open(partial, "wb") as handle:
                for chunk in response.iter_bytes(CHUNK_SIZE):
                    handle.write(chunk)
                    written += len(chunk)
        except httpx.HTTPError as exc:
            raise transport_failure(exc, request_id) from exc
        _check_complete(response, written, request_id)
        os.replace(partial, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            partial.unlink()
    return DownloadedFile(
        path=target,
        size=written,
        content_type=response.headers.get("content-type", ""),
        filename=filename,
        request_id=request_id,
    )


async def awrite_stream(
    response: httpx.Response, path: PathLike, *, filename: Optional[str] = None
) -> DownloadedFile:
    """Async twin of :func:`write_stream`. Writes run on a worker thread."""
    request_id = response.headers.get("x-request-id")
    target = resolve_target(path, filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_for(target)
    written = 0
    try:
        try:
            with open(partial, "wb") as handle:
                async for chunk in response.aiter_bytes(CHUNK_SIZE):
                    await asyncio.to_thread(handle.write, chunk)
                    written += len(chunk)
        except httpx.HTTPError as exc:
            raise transport_failure(exc, request_id) from exc
        _check_complete(response, written, request_id)
        os.replace(partial, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            partial.unlink()
    return DownloadedFile(
        path=target,
        size=written,
        content_type=response.headers.get("content-type", ""),
        filename=filename,
        request_id=request_id,
    )
