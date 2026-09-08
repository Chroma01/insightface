"""Exercise shipped health probes and documented setup without Docker or sudo."""

from __future__ import annotations

import json
import os
import re
import shlex
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.response import addinfourl

import pytest

SERVER_DIR = Path(__file__).resolve().parents[2]
SETUP_GUIDES = sorted(SERVER_DIR.glob("README*.md")) + sorted(
    (SERVER_DIR / "docs").glob("user-guide*.md")
)


@pytest.mark.parametrize("variant", ["cpu", "cuda12"])
@pytest.mark.parametrize("outcome", [200, 503, "connection_error"])
def test_image_healthcheck_bypasses_proxies_and_preserves_failures(
    monkeypatch: pytest.MonkeyPatch, variant: str, outcome: int | str,
) -> None:
    dockerfile = (SERVER_DIR / "docker" / f"Dockerfile.{variant}").read_text()
    healthcheck = re.search(r"HEALTHCHECK[^\n]+\\\n\s+CMD (\[[^\n]+\])", dockerfile)
    assert healthcheck is not None
    command = json.loads(healthcheck[1])
    assert command[:2] == ["python", "-c"]
    assert len(command) == 3

    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(variable, "http://unreachable-proxy.invalid:3128")
    for variable in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(variable, "")
    monkeypatch.delenv("REQUEST_METHOD", raising=False)
    # Do not let macOS system proxy exclusions hide a regression on Linux.
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)
    monkeypatch.setattr(urllib.request, "_opener", None)

    requests = []

    def offline_http_open(self, request):
        requests.append(request)
        assert request.host == "127.0.0.1:8080"
        assert request.selector == "/v1/health"
        assert not request.has_proxy()
        assert request.timeout == 3
        if outcome == "connection_error":
            raise urllib.error.URLError("offline simulated connection refusal")
        response = addinfourl(BytesIO(b'{"status":"ready"}'), Message(), request.full_url, outcome)
        response.msg = "OK" if outcome == 200 else "Service Unavailable"
        return response

    def no_network(*args, **kwargs):
        raise AssertionError("healthcheck regression tests must stay offline")

    # Keep urllib's real proxy and HTTP status processing; replace only transport.
    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", offline_http_open)
    monkeypatch.setattr(socket, "create_connection", no_network)
    if outcome == 200:
        exec(command[2], {})
    else:
        expected = urllib.error.URLError if outcome == "connection_error" else urllib.error.HTTPError
        with pytest.raises(expected):
            exec(command[2], {})
    assert len(requests) == 1


@pytest.mark.parametrize("guide", SETUP_GUIDES, ids=lambda path: path.name)
@pytest.mark.parametrize("existing_addons", [False, True], ids=["empty-checkout", "existing-directory"])
def test_initial_setup_prepares_addons_before_any_compose_command(
    tmp_path: Path, guide: Path, existing_addons: bool,
) -> None:
    markdown = guide.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", markdown, re.DOTALL)
    setup = next(block for block in blocks if "docker compose" in block)
    assert "mkdir" in setup
    if existing_addons:
        directory = tmp_path / "server" / ".models" / "addons"
        directory.mkdir(parents=True)
        directory.chmod(0o700)
        directory.parent.chmod(0o700)

    # Validate actual mkdir/chgrp/chmod and shell exports at the first Docker
    # invocation. No container or privileged command is run. The one privileged
    # operation (assigning GID 10001) is mapped to the test user's own group;
    # its requested shared GID is still checked below.
    probe = tmp_path / "check_setup.py"
    probe.write_text(
        """import os
from pathlib import Path
root = Path('server/.models')
addons = root / 'addons'
assert addons.is_dir(), 'addon bind source must exist before Compose'
assert root.stat().st_mode & 0o005 == 0o005, 'Server must be able to traverse /models'
assert addons.stat().st_mode & 0o2070 == 0o2070, 'shared group needs rwx and setgid'
assert addons.stat().st_gid == os.getgid()
assert os.environ.get('INSIGHTFACE_MODELS_UID') == str(os.getuid())
assert os.environ.get('INSIGHTFACE_MODELS_GID') == str(os.getgid())
assert Path('shared-group-requested').read_text().strip() == '10001'
temporary = addons / '.write-probe'
temporary.write_text('synthetic addon')
assert temporary.stat().st_gid == addons.stat().st_gid
temporary.unlink()
print('prepared-before-compose')
""",
        encoding="utf-8",
    )
    script = """umask 077
sudo() {
  if [ "$1" = chgrp ]; then
    test "$2" = 10001
    printf '%s\\n' "$2" > shared-group-requested
    shift 2
    command chgrp "$(id -g)" "$@"
  else
    test "$1" = chmod
    command "$@"
  fi
}
curl() { :; }
"""
    script += f"docker() {{ {shlex.quote(sys.executable)} {shlex.quote(str(probe))}; }}\n"
    script += setup
    environment = os.environ.copy()
    environment.pop("INSIGHTFACE_MODELS_UID", None)
    environment.pop("INSIGHTFACE_MODELS_GID", None)
    result = subprocess.run(
        ["bash", "-euc", script], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, f"{guide.name}: {result.stderr}"
    assert "prepared-before-compose" in result.stdout
