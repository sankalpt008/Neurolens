import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurolens.bench.env import collect_env  # noqa: E402
from neurolens.bench.env import collect_env


def test_collect_env_returns_required_keys():
    env = collect_env()
    for key in {"driver_version", "cuda_version", "gpu_name", "os", "python_version"}:
        assert key in env
        assert isinstance(env[key], str)
        assert env[key] is not None
