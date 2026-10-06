"""Restore the provided trained policies at their original relative paths."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads((ROOT / "artifacts" / "manifest.json").read_text(encoding="utf-8"))
    grouped = {}
    for record in manifest["archived_models"]:
        grouped.setdefault(record["archive"], []).append(record)
    restored = unchanged = 0
    for archive_name, records in grouped.items():
        with zipfile.ZipFile(ROOT / archive_name) as archive:
            for record in records:
                relative = PurePosixPath(record["path"])
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "runs":
                    raise ValueError("Invalid archive destination")
                target = ROOT.joinpath(*relative.parts).resolve()
                if not target.is_relative_to(ROOT.resolve()):
                    raise ValueError("Archive destination is outside the repository")
                if target.exists():
                    if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
                        raise RuntimeError(f"Preserving a different local file: {record['path']}")
                    unchanged += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".restoring")
                try:
                    with archive.open(record["path"]) as source, temporary.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    if hashlib.sha256(temporary.read_bytes()).hexdigest() != record["sha256"]:
                        raise RuntimeError(f"Artifact checksum mismatch: {record['path']}")
                    temporary.replace(target)
                    restored += 1
                finally:
                    temporary.unlink(missing_ok=True)
    print(json.dumps({"restored": restored, "already_present": unchanged}))


if __name__ == "__main__":
    main()
