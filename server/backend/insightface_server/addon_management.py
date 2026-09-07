"""Explicit Web addon preparation; running inference never changes here."""

from __future__ import annotations

import asyncio
import fcntl
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import tomlkit
from fastapi import Request

from .addons import install_addon
from .config import Settings, load_server_config
from .errors import ApiError


def require_management_request(request: Request, cors_origins: tuple[str, ...]) -> None:
    """Require a non-simple JSON request and reject unrelated browser origins."""

    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise ApiError("json_required", "Use Content-Type: application/json with an empty object.", 415)
    origin = request.headers.get("origin")
    if origin is None:
        return  # CLI/API clients do not send Origin.
    try:
        source = urlsplit(origin)
    except ValueError:
        raise ApiError("origin_not_allowed", "This origin may not change Server configuration.", 403) from None
    valid_origin = (
        origin != "null"
        and source.scheme in ("http", "https")
        and bool(source.netloc)
        and not source.path
        and not source.query
        and not source.fragment
        and source.username is None
    )
    same_origin = source.netloc == request.url.netloc and source.scheme == request.url.scheme
    if not valid_origin or (not same_origin and origin not in cors_origins and "*" not in cors_origins):
        raise ApiError("origin_not_allowed", "This origin may not change Server configuration.", 403)


def _writable(path: Path, *, directory: bool = False) -> bool:
    try:
        mode = path.stat().st_mode
        # Check permission bits too: tests or a root container must not mistake
        # a deliberately read-only file for a supported editable deployment.
        return bool(mode & 0o222) and os.access(path, os.W_OK | (os.X_OK if directory else 0))
    except OSError:
        return False


def _file_mount(path: Path) -> bool:
    if os.path.ismount(path):
        return True
    # os.path.ismount does not detect same-filesystem Linux bind mounts.
    try:
        target = str(path.resolve())
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            mount_point = line.split()[4]
            for escaped, literal in ((r"\040", " "), (r"\011", "\t"), (r"\012", "\n"), (r"\134", "\\")):
                mount_point = mount_point.replace(escaped, literal)
            if mount_point == target:
                return True
    except (OSError, IndexError):
        pass
    return False


class LivenessManager:
    def __init__(self, settings: Settings, *, enabled: bool):
        self.settings = settings
        self.enabled = enabled
        self.model_path = settings.models_dir / "addons" / "liveness.onnx"
        self._mutex = threading.Lock()
        self._state = "idle"
        self._error: dict[str, str] | None = None
        self._task: asyncio.Task | None = None
        self._stopping = threading.Event()

    def _installed(self) -> tuple[bool, dict[str, str] | None]:
        from insightface.addons import ensure_addon

        try:
            ensure_addon("liveness", root=self.settings.models_dir, download=False)
            return True, None
        except FileNotFoundError:
            return False, None
        except (OSError, RuntimeError):
            return False, {
                "code": "addon_model_invalid",
                "message": (
                    f"The existing addon at {self.model_path} is unreadable or failed SHA256 "
                    "verification. Restore the official model, or remove the invalid file "
                    "and retry. It will not be overwritten automatically."
                ),
            }

    def _capability(self) -> tuple[str, str] | None:
        path = self.settings.config_file
        if path is None:
            return "config_file_missing", "Set INSIGHTFACE_CONFIG_FILE to an editable server.toml to enable liveness from the Web UI."
        if not path.is_file() or path.is_symlink():
            return "config_file_not_regular", "The configured server.toml must be an existing regular file, not a symbolic link."
        if _file_mount(path):
            return "config_file_mount", "Mount the configuration directory writable instead of bind-mounting the single server.toml file, then recreate the container."
        if not _writable(path) or not _writable(path.parent, directory=True):
            return "config_not_writable", "The configuration file and its directory must be writable by the Server user (UID 10001 in the supplied Docker image). Grant host directory/file permissions, use a writable configuration directory mount, and recreate the container."
        addon_dir = self.model_path.parent
        parent = addon_dir if addon_dir.exists() else self.settings.models_dir
        if not parent.is_dir() or not _writable(parent, directory=True):
            return "addon_directory_not_writable", "Mount the addon directory writable at /models/addons and grant host directory permissions to the Server user (UID 10001 in the supplied Docker image); base models may remain read-only. Recreate the container after changing mounts."
        return None

    def status(self) -> dict[str, Any]:
        installed, artifact_error = self._installed()
        capability = self._capability()
        unavailable_code, reason = capability if capability else (None, None)
        configured_enabled = self.enabled
        config_error = None
        try:
            configured_enabled = "liveness" in load_server_config(self.settings.config_file).addons
        except ValueError:
            reason = "The current server.toml is unreadable or invalid. Correct it before enabling liveness."
            unavailable_code = "addon_config_invalid"
            config_error = {"code": "addon_config_invalid", "message": reason}
        with self._mutex:
            state, error = self._state, self._error
        error = error or config_error or artifact_error
        if reason is None and artifact_error:
            unavailable_code, reason = artifact_error["code"], artifact_error["message"]
        if reason is None and self._stopping.is_set():
            unavailable_code, reason = "server_stopping", "The Server is shutting down."
        if state != "downloading":
            if error:
                state = "error"
            else:
                state = "ready" if installed else "idle"
        return {
            "enabled": self.enabled,
            "installed": installed,
            "configured_enabled": configured_enabled,
            "restart_required": configured_enabled != self.enabled,
            "can_enable": reason is None and artifact_error is None and not self._stopping.is_set(),
            "unavailable_code": unavailable_code,
            "unavailable_reason": reason,
            "state": state,
            "error": error,
            "model_path": str(self.model_path),
            "config_file": str(self.settings.config_file) if self.settings.config_file else None,
        }

    async def enable(self) -> dict[str, Any]:
        snapshot = await asyncio.to_thread(self.status)
        if not snapshot["can_enable"]:
            raise ApiError(
                "addon_management_unavailable",
                snapshot["unavailable_reason"] or "The Server is shutting down.",
                409,
            )
        # No await between checking and publishing the owned task: duplicate
        # requests on this event loop join the same job.
        if self._task is None or self._task.done():
            with self._mutex:
                self._state, self._error = "downloading", None
            self._task = asyncio.create_task(asyncio.to_thread(self._prepare))
        return await asyncio.to_thread(self.status)

    def _save_config(self, path: Path) -> None:
        # Caller holds an advisory lock across the complete preparation job.
        # Re-read after the download so unrelated intervening edits survive.
        original = path.read_text(encoding="utf-8")
        load_server_config(path)
        document = tomlkit.parse(original)
        for section, key in (("inference", "addons"), ("addons", "auto_download")):
            if section not in document:
                document[section] = tomlkit.table()
            table = document[section]
            if key not in table:
                table[key] = tomlkit.array()
            if "liveness" not in table[key]:
                table[key].append("liveness")
        updated = tomlkit.dumps(document)
        if original == updated:
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=".server-config-", suffix=".toml", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
                stream.write(updated)
                stream.flush()
                os.fsync(stream.fileno())
            load_server_config(temporary)
            if self._stopping.is_set():
                raise RuntimeError("Server shutdown interrupted addon preparation")
            if path.read_text(encoding="utf-8") != original:
                raise RuntimeError("Configuration changed while saving; retry")
            os.replace(temporary, path)
            temporary = None
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _prepare(self) -> None:
        stage = "config"
        try:
            path = self.settings.config_file
            assert path is not None
            # A stable sibling lock survives atomic config replacement. Sharing
            # the configuration directory across processes shares this lock.
            lock_fd = os.open(path.parent / ".liveness-management.lock", os.O_CREAT | os.O_RDONLY, 0o644)
            with os.fdopen(lock_fd, "r") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ApiError("addon_job_in_progress", "Another Server is preparing liveness; wait and refresh.", 409) from None
                capability = self._capability()
                if capability:
                    raise ApiError("addon_management_unavailable", capability[1], 409)
                load_server_config(path)
                stage = "download"
                installed, artifact_error = self._installed()
                if artifact_error:
                    raise ApiError(artifact_error["code"], artifact_error["message"], 409)
                if not installed:
                    install_addon("liveness", self.settings.models_dir)
                installed, artifact_error = self._installed()
                if not installed:
                    raise RuntimeError("Addon verification failed after installation")
                stage = "config"
                self._save_config(path)
            with self._mutex:
                self._state, self._error = "ready", None
        except Exception as exc:
            # Requests exceptions can contain proxy user/passwords. Never copy
            # their text to the public status or logs.
            error = (
                {"code": exc.code, "message": exc.message}
                if isinstance(exc, ApiError)
                else {
                    "code": "addon_download_failed" if stage == "download" else "addon_config_save_failed",
                    "message": (
                        "Could not download or verify the official liveness model. Check the Server network/proxy settings and retry. Configuration was not changed."
                        if stage == "download"
                        else "Could not save the liveness configuration. Check server.toml, directory mount permissions, and concurrent edits, then retry. A downloaded model can be reused."
                    ),
                }
            )
            with self._mutex:
                self._state, self._error = "error", error

    async def close(self) -> None:
        self._stopping.set()
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=10)
            except TimeoutError:
                # The downloader has bounded connect/read timeouts. Its worker
                # may finish caching a file but must not save config afterward.
                pass
