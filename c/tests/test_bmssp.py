import subprocess
from pathlib import Path

def test_run_sssp():
    root = Path(__file__).resolve().parent.parent
    src = root / "bmssp.c"
    exe = root / "bmssp"
    subprocess.check_call(["gcc", str(src), "-lm", "-o", str(exe)])
    out = subprocess.check_output([str(exe)], text=True).strip().splitlines()
    assert out[0].strip().startswith("0")
    assert out[1].strip().startswith("1")
    assert out[2].strip().startswith("4")
    assert out[3].strip().lower().startswith("inf")
