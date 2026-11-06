"""Verify the companion's text scope, manuscript, and archived numerical claims."""
from hashlib import sha256
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper" / "observation-age" / "observation-age-stopping-support-paper.pdf"
PAPER_SHA256 = "bc8cb11414093f3c2e8ce7f1e83d773906a4e6e6aa1f71c1817a3235a6827198"
HAN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U000323af]")
UNPUBLISHED_PATH = re.compile(r"/home/xiaotwu/|/mnt/c/Users/|[Cc]:[/\\]Users[/\\]", re.IGNORECASE)


def audit():
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", "build"}
    for item in ROOT.rglob("*"):
        relative = item.relative_to(ROOT)
        if excluded.intersection(relative.parts) or not item.is_file():
            continue
        assert not HAN.search(relative.as_posix()), relative
        if item.suffix in {".pdf", ".png"}:
            continue
        text = item.read_text(encoding="utf-8")
        assert not HAN.search(text), relative
        if relative.name != Path(__file__).name:
            assert not UNPUBLISHED_PATH.search(text), relative
    assert sha256(PAPER.read_bytes()).hexdigest() == PAPER_SHA256
    for command in (["evidence/verify_evidence.py"], ["replay/selfcheck.py"], ["-m", "unittest", "risk.test_risk"]):
        subprocess.run([sys.executable, "-B", *command], cwd=ROOT, check=True)
    print("English text, paper SHA-256, replay self-check, and risk/evidence tests passed.")


if __name__ == "__main__":
    audit()

