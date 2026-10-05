"""Build a dependency-free, inspectable source distribution without personal runtime data."""
import hashlib
import json
import zipfile
from pathlib import Path

if __package__:
    from .public_evidence import redact
else:
    from public_evidence import redact


def build(root, destination):
    root = Path(root)
    files = []
    for directory in ("optchat","scripts","tests"):
        files.extend(p for p in (root/directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix!=".pyc")
    files.extend(root/name for name in ("README.md","LICENSE","config.example.yaml",".gitignore"))
    files.extend(root/"evidence"/name for name in (
        "gist-metadata.json","hermes-wire-report.json","hermes-wire.jsonl",
        "live-quality-report.json","live-wire.jsonl","installation-report.json","release-readiness.json","archive-example.html",
        "cache-comparison-report.json","cache-latest.json","cache-native-report.json","cache-native-wire.jsonl"))
    for run in ("cache-c365f7903b014e93ac3e09d900f559dd","cache-638688684b8b41078a82611b3e99d973"):
        files.extend(root/"evidence"/run/name for name in ("report.json","results.json","wire.jsonl"))
    files = sorted(files)
    contents = {}
    for path in files:
        data = path.read_bytes()
        if path.is_relative_to(root/"evidence"):
            data = redact(data.decode("utf-8")).encode("utf-8")
        contents[str(path.relative_to(root))] = data
    manifest = {name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        contents["manifest.json"] = (json.dumps(manifest,indent=2)+"\n").encode()
        for name,data in contents.items():
            member = zipfile.ZipInfo("optchat-hermes/"+name)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member,data)
    return destination


def main():
    root = Path(__file__).resolve().parents[1]
    # plugin.yaml is the single version source; Hermes reads the same field.
    version = next(line.split(":",1)[1].strip() for line in (root/"optchat"/"plugin.yaml").read_text().splitlines()
                   if line.startswith("version:"))
    destination = build(root,root/"dist"/f"optchat-hermes-{version}.zip")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    (destination.parent/"SHA256SUMS").write_text(digest+"  "+destination.name+"\n")
    print(destination)
    print("sha256: "+digest)


if __name__ == "__main__":
    main()
