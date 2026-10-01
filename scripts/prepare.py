import hashlib
import shutil
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import py7zr


def find_project_root(marker: str = "pyproject.toml") -> Path:
    path = Path.cwd().resolve()
    while not (path / marker).exists():
        if path.parent == path:
            raise FileNotFoundError(f"{marker!r} not found above {Path.cwd()}")
        path = path.parent
    return path


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def extract_checked(archive_path: Path, extract_dir: Path) -> None:
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # First extract archive into isolation
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(tmp_dir)

        # Then merge into final extraction directory
        for src in sorted(tmp_dir.rglob("*")):
            if not src.is_file():
                continue

            rel_path = src.relative_to(tmp_dir)
            dst = extract_dir / rel_path

            if dst.exists():
                if sha256(src) == sha256(dst):
                    print(f"SKIP identical: `{rel_path}`")
                    continue

                warnings.warn(
                    f"Hash mismatch for `{rel_path!s}`\n  existing: `{dst}`\n  archive: `{archive_path.name}`",
                    stacklevel=2,
                )
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)

            print(f"EXTRACT: {rel_path}")


PROJECT_ROOT: Final[Path] = find_project_root()
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
EXTRACT_DIR: Final[Path] = RAW_DIR / "extracted"


if __name__ == "__main__":
    if not RAW_DIR.exists():
        raise OSError(f"Folder `{RAW_DIR}` not available!")

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    for path in RAW_DIR.glob("*.7z"):
        print(f"\n{path.name}")
        extract_checked(path, EXTRACT_DIR)
