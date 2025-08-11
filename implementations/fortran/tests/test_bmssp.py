import subprocess
from pathlib import Path


def test_run_sssp():
    root = Path(__file__).resolve().parent.parent
    src = root / "bmssp.f90"
    exe = root / "bmssp"
    subprocess.check_call(["gfortran", str(src), "-o", str(exe)])
    out = subprocess.check_output([str(exe)], text=True).strip().splitlines()
    assert out[0].strip().startswith("0")
    assert out[1].strip().startswith("1")
    assert out[2].strip().startswith("3")
    assert out[3].strip().startswith("4")
