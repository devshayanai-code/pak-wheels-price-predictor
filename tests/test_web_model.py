"""The website runs the model in JavaScript; check it gives Python's exact answers."""
import json
import shutil
import subprocess

import pytest

from src.config import ROOT_DIR

DOCS = ROOT_DIR / "docs"

CHECK_SCRIPT = """
const P = require("./docs/predictor.js");
const model = require("./docs/model.json");
const ref = require("./docs/reference.json");
const worst = Math.max(...ref.cars.map((car, i) =>
  Math.abs(P.predictRaw(model, car) - ref.prices[i]) / ref.prices[i]));
console.log(JSON.stringify({ worst }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_javascript_matches_python():
    result = subprocess.run(
        ["node", "-e", CHECK_SCRIPT], cwd=ROOT_DIR, capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout)["worst"] < 1e-9


def test_site_files_exist():
    for name in ["index.html", "predictor.js", "model.json", "options.json", "listings.json"]:
        assert (DOCS / name).exists(), f"docs/{name} is missing: run python -m src.export_web"
