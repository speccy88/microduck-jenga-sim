"""Run the standalone CPU checks without installing a test runner."""
import json
import os
from pathlib import Path
import runpy
import time
import traceback

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("Run with CUDA_VISIBLE_DEVICES='' for this CPU-only validation.")
root = Path(__file__).resolve().parents[1]
checks = runpy.run_path(str(root / "tests/test_jenga.py"))
results = []
for name, check in checks.items():
    if name.startswith("test_") and callable(check):
        start = time.monotonic()
        result = {"name": name}
        try:
            check()
            result["passed"] = True
        except Exception:
            result.update(passed=False, traceback=traceback.format_exc())
        result["seconds"] = round(time.monotonic()-start, 3)
        results.append(result)
        print(json.dumps(result), flush=True)
out = root / "evidence"
out.mkdir(exist_ok=True)
(out / "cpu_validation.json").write_text(json.dumps(results, indent=2)+"\n")
raise SystemExit(0 if all(r["passed"] for r in results) else 1)
