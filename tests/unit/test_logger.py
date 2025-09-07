import json
import sys

from ssspx.logger import StdLogger


def test_stdlogger_plain_and_json(capsys):
    log = StdLogger(level="debug")
    log.info("event", answer=42)
    log.debug("dbg")
    err = capsys.readouterr().err
    assert "event" in err and "answer=42" in err

    log_json = StdLogger(level="info", json_fmt=True, stream=sys.stdout)
    log_json.info("run", foo="bar")
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data["event"] == "run" and data["foo"] == "bar"
