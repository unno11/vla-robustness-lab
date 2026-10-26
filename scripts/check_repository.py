"""Check repository syntax, documentation links and initial result consistency."""
import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "__pycache__", ".venv", "data", "datasets", "checkpoints", "weights",
        "cache", "artifacts", "logs", "raw"}
errors = []
checked = 0
for path in sorted(ROOT.rglob("*")):
    relative = path.relative_to(ROOT)
    if not path.is_file() or set(relative.parts) & SKIP:
        continue
    if path.name.endswith(".local.json"):
        continue
    checked += 1
    try:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".md":
            body = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
            for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", body):
                if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                    continue
                target = unquote(target.split("#", 1)[0].split("?", 1)[0])
                if target and not (path.parent / target).exists():
                    errors.append(f"{relative}: missing local link {target}")
    except (SyntaxError, ValueError, UnicodeError) as error:
        errors.append(f"{relative}: {error}")

status = json.loads((ROOT / "results/status.json").read_text(encoding="utf-8"))
if not status["benchmark_completed"] and status["success_rate"] is not None:
    errors.append("Incomplete benchmark must not claim a full-suite success rate")
if not status["policy_rollout_completed"] and status["successes"] is not None:
    errors.append("No rollout completed, but successes have been populated")
if errors:
    raise SystemExit("\n".join(errors))
print(f"PASS: {checked} repository files; Python/JSON syntax, local documentation links and initial result consistency")
print("Scope: static repository checks only; no policy, simulator or GPU validation")
