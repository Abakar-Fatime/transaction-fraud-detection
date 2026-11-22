import argparse
import json

from fraud.pipeline import run

p = argparse.ArgumentParser(prog="fraud")
sub = p.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("run", help="train, evaluate and export everything")
r.add_argument("--data", required=True)
r.add_argument("--quick", action="store_true")
n = sub.add_parser("notes", help="write grounded analyst notes for the top alerts with a local Ollama model")
n.add_argument("--model", default="qwen2.5:7b")
n.add_argument("--host", default="http://localhost:11434")
n.add_argument("--limit", type=int, default=60)
n.add_argument("--console", default="console/data/console.json")
n.add_argument("--out", default="console/data/notes.json")
a = p.parse_args()
if a.cmd == "run":
    run(a.data, quick=a.quick)
elif a.cmd == "notes":
    from pathlib import Path
    from fraud import notes
    alerts = json.loads(Path(a.console).read_text())["alerts"][: a.limit]
    res = notes.generate(alerts, a.model, a.host)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(res["stats"])
