"""Ensure repository root is on ``sys.path`` for ``import advsm``."""

from advsm._paths import ensure_repo_on_path, ensure_third_party_on_path

ensure_repo_on_path()
ensure_third_party_on_path()
