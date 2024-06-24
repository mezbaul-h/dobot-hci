# pylint: disable=wrong-import-position
import os
import sys
import typing
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPOSITORY_ROOT.resolve()))

import re
from urllib.parse import urljoin

import requests

from sample_package import __version__

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_API_URL = os.getenv("GITHUB_API_URL", "")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "")
PROJECT_NAME = GITHUB_REPOSITORY.split("/")[-1]
DEFAULT_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
}


def make_release_body() -> str:
    changelog_content = (REPOSITORY_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    version_pat = re.compile(r"^## \[([^]]+)] - \d{4}-\d{2}-\d{2}")
    lines: typing.Optional[typing.List[str]] = None

    for line in changelog_content.splitlines():
        mobj = version_pat.match(line)

        if lines is None and mobj and mobj.group(1) == __version__:
            lines = []
        elif lines is not None and mobj and mobj.group(1) != __version__:
            break
        elif lines is not None:
            lines.append(line)

    return "\n".join(lines).strip() if lines else ""


def release_exists() -> bool:
    api_path = f"/repos/{GITHUB_REPOSITORY}/releases/tags/v{__version__}"
    url = urljoin(GITHUB_API_URL, api_path)
    res = requests.get(url, headers=DEFAULT_HEADERS, timeout=5)

    if not res.ok:
        return False

    return True


def release():
    if not release_exists():
        body = {
            "tag_name": f"v{__version__}",
            "target_commitish": "master",
            "name": f"{PROJECT_NAME} {__version__}",
            "draft": False,
            "body": make_release_body(),
            "prerelease": "pre" in __version__,
        }

        api_path = f"/repos/{GITHUB_REPOSITORY}/releases"
        url = urljoin(GITHUB_API_URL, api_path)

        requests.post(url, headers=DEFAULT_HEADERS, json=body, timeout=5)


if __name__ == "__main__":
    release()
