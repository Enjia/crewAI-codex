from pathlib import Path


def _is_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _find_local_repo_root(start: Path) -> Path | None:
    if start.is_file():
        start = start.parent

    for parent in [start, *start.parents]:
        if (parent / "lib" / "crewai").is_dir() and (
            parent / "lib" / "crewai-tools"
        ).is_dir():
            return parent
    return None


def _resolve_local_repo_root(explicit_path: str | None) -> Path | None:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        return None

    cwd_root = _find_local_repo_root(Path.cwd().resolve())
    if cwd_root:
        return cwd_root

    file_root = _find_local_repo_root(Path(__file__).resolve())
    if file_root:
        return file_root

    return None


def _build_uv_sources(repo_root: Path) -> dict[str, Path]:
    candidates = {
        "crewai": repo_root / "lib" / "crewai",
        "crewai-tools": repo_root / "lib" / "crewai-tools",
        "crewai-files": repo_root / "lib" / "crewai-files",
    }

    return {name: path for name, path in candidates.items() if path.is_dir()}


def _append_uv_sources(pyproject_path: Path, sources: dict[str, Path]) -> bool:
    if not sources:
        return False

    if not pyproject_path.exists():
        return False

    content = pyproject_path.read_text()
    if "[tool.uv.sources]" in content:
        return False

    lines = ["", "[tool.uv.sources]"]
    for name, path in sources.items():
        lines.append(
            f'{name} = {{ path = "{path.as_posix()}", editable = true }}'
        )

    pyproject_path.write_text(content.rstrip() + "\n" + "\n".join(lines) + "\n")
    return True
