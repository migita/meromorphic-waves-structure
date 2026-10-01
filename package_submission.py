#!/usr/bin/env python3
"""Package the current built paper and certificates without uploading them.

Run `make archives` to rebuild and check first. This script uses an explicit
manifest to exclude research notes, temporary files, and Git history.
It requires only the Python standard library.
"""

import gzip
import hashlib
import io
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent
ANCILLARY = (
    "README.md",
    "requirements.txt",
    "check_bracket.py",
    "check_bracket_audit.py",
    "check_bracket_audit_certificate.json",
    "check_bracket_certificate.json",
    "check_drift_n2.py",
    "check_examples.py",
    "check_ks_family.py",
    "check_ks_residue.py",
    "check_linear_terms.py",
    "check_more_examples.py",
    "check_sharpness.py",
    "check_text_identities.py",
    "ks_residue_checks.json",
    "linear_terms_checks.json",
    "more_examples_checks.json",
    "network_32_explicit.py",
    "run_all.py",
    "sectors.py",
    "series_tools.py",
    "sh_dispersion.py",
    "sharpness_checks.json",
    "tables_q2.py",
    "text_identities_checks.json",
)
SOURCE = ("main.tex", "main.bbl", "references.bib") + tuple(
    "anc/" + name for name in ANCILLARY
)
BUNDLE = SOURCE + (
    "main.pdf", "README.md", "Makefile", "package_submission.py", ".gitignore"
)


def main():
    for name in BUNDLE:
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"Missing or invalid file: {name}. Run make archives.")

    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    arxiv = destination / "arxiv-source.tar.gz"
    zenodo = destination / "zenodo-bundle.zip"

    # Fixed archive metadata makes packaging identical input files reproducible.
    with arxiv.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name in sorted(SOURCE):
                    data = (ROOT / name).read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO(data))

    with zipfile.ZipFile(zenodo, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(BUNDLE):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT / name).read_bytes(),
                             compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    lines = []
    for path, count in ((arxiv, len(SOURCE)), (zenodo, len(BUNDLE))):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}\n")
        print(f"{path.relative_to(ROOT)}: {count} files, {path.stat().st_size:,} bytes")
    (destination / "SHA256SUMS.txt").write_text("".join(lines))
    print("Checksums: dist/SHA256SUMS.txt")


if __name__ == "__main__":
    main()
