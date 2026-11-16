"""The Qali workspace: the folder a run reads its inputs from and writes into.

One rule for the GUI and the CLI: the selected project folder if there is
one, else Qali's per-user data home. ``Workspace.current()`` (the single
accessor every use-time site resolves through) is the pinned *active*
workspace (:func:`set_active`) when one is set, otherwise
:func:`default_workspace` — the folder named by ``QALI_PROJECT_ROOT``, else
the per-user home. The launch directory never matters.

Constructing a :class:`Workspace` does no IO: directory *locations* resolve on
access. :func:`seed_bundled_data` fills a workspace from the bundled
resources without ever overwriting a file, and skips what the user removed
(see :func:`mark_removed`).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qali.core.fileio import is_safe_name, write_json
from qali.core.paths import (
    project_root_from_env,
    reference_path,
    runtime_root_for,
    user_data_root,
)
from qali.core.species import is_ion_id

# Bundled trees copied into a workspace: resources/<tree> -> workspace dir.
# Pipeline inputs are seeded for every run; the shipped example results are
# only seeded where a browsable folder is wanted (the GUI), so a fresh folder
# opens with populated species instead of an empty list.
_INPUT_TREES = {"data_json": "data_json_dir"}
_EXAMPLE_TREES = {"plots": "plots_dir", "orca_outputs": "orca_outputs_dir"}

# The per-species parts of a workspace, by the name of their bundled tree.
SPECIES_TREES = ("data_json", "plots", "orca_outputs", "tables")
# Remembers species the user removed from this folder (see mark_removed).
REMOVED_SPECIES_FILE = "removed_species.json"
# Per seeded tree: which files are untouched copies of the bundle, so a newer bundle can replace them.
SEEDED_FILE = ".qali_seeded.json"
# Where an older copy of a bundled input file goes when the bundle's replaces it.
PREVIOUS_DIR = ".qali_previous"
# data_json files that are inputs: no stage writes them, they come from the bundle (literature data).
_INPUT_FILE = re.compile(r"(^|/)([^/]+_(feshbach|hf_constants|feshbach_table|nist)\.json|nuclear_data\.json)$")


@dataclass(frozen=True)
class Workspace:
    """An output workspace rooted at a project folder."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser())

    @property
    def runtime_root(self) -> Path:
        """Where outputs land under *root* (``<root>/.qali`` for a checkout)."""
        return runtime_root_for(self.root)

    @property
    def data_json_dir(self) -> Path:
        """Writable JSON data directory."""
        return self.runtime_root / "data"

    @property
    def plots_dir(self) -> Path:
        """Generated HTML plots directory."""
        return self.runtime_root / "plots"

    @property
    def orca_outputs_dir(self) -> Path:
        """Raw ORCA output runs directory."""
        return self.runtime_root / "orca_outputs"

    @property
    def basis_sets_dir(self) -> Path:
        """Bundled, read-only basis sets (independent of the root)."""
        return reference_path("Basis sets")

    def output_path(self, relative: str) -> Path:
        """Resolve a runtime-relative template such as ``data/{el}.json``."""
        return self.runtime_root / relative

    @classmethod
    def current(cls) -> Workspace:
        """The workspace in effect right now: the pinned one, else the default."""
        return active() or default_workspace()


_ACTIVE: Workspace | None = None


def set_active(workspace: Workspace | None) -> None:
    """Pin *workspace* as current (per job run or test); ``None`` unpins."""
    global _ACTIVE
    _ACTIVE = workspace


def active() -> Workspace | None:
    """The pinned workspace, if any."""
    return _ACTIVE


def default_workspace() -> Workspace:
    """The workspace the configuration implies when none is pinned.

    The folder named by ``QALI_PROJECT_ROOT`` (the GUI sets it for every
    stage subprocess; it can be set from a shell too), else the per-user
    data home.
    """
    configured = project_root_from_env()
    if configured:
        return Workspace(root=Path(configured).expanduser().resolve())
    return Workspace(root=user_data_root())


def stem_belongs_to_species(species: str, stem: str) -> bool:
    """True if a data file's *stem* is *species*'s: ``Na``, ``Na_lifetimes``.

    ``Na_c1_lifetimes`` is the ion's file, so it does not belong to ``Na``.
    """
    if stem == species:
        return True
    if not stem.startswith(f"{species}_"):
        return False
    return not is_ion_id(f"{species}_{stem[len(species) + 1:].split('_', 1)[0]}")


def _species_data_file(species: str, relative: Path) -> bool:
    """True if *relative*, under the data folder, is one of *species*'s files."""
    in_data_or_arc = len(relative.parts) == 1 or (
        len(relative.parts) == 2 and relative.parts[0] == "arc")
    return (in_data_or_arc and relative.suffix == ".json"
            and stem_belongs_to_species(species, relative.stem))


def species_files(workspace: Workspace, species: str) -> dict[str, list[Path]]:
    """The paths holding *species*'s results, by tree in :data:`SPECIES_TREES`.

    Data is listed file by file, because the folder is shared; plots and ORCA
    runs are the species' own folder. Shared cross-element files are never
    included. A name that is not a plain file name matches nothing.
    """
    found: dict[str, list[Path]] = {tree: [] for tree in SPECIES_TREES}
    if not is_safe_name(species):
        return found
    data_dir = workspace.data_json_dir
    if data_dir.is_dir():
        found["data_json"] = sorted(
            path for path in data_dir.rglob("*.json")
            if _species_data_file(species, path.relative_to(data_dir)))
    mine = workspace.runtime_root / "tables" / "files"               # files the user imported or edited
    if mine.is_dir():
        # By file name, not path: the folder names sort differently on Windows and Linux.
        found["tables"] = sorted((p for p in mine.rglob("*")
                                  if p.is_file() and stem_belongs_to_species(species, p.stem)),
                                 key=lambda p: (p.name.lower(), str(p)))
    for tree, folder in (("plots", workspace.plots_dir),
                         ("orca_outputs", workspace.orca_outputs_dir)):
        if (folder / species).is_dir():
            found[tree] = [folder / species]
    return found


def _removed_state(workspace: Workspace) -> dict:
    """The folder's removed-species record; empty if absent or unreadable."""
    try:
        text = (workspace.runtime_root / REMOVED_SPECIES_FILE).read_text(encoding="utf-8")
        state = json.loads(text)
    except (OSError, ValueError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    hidden = state.get("hidden")
    no_seed = state.get("no_seed")
    return {
        "hidden": hidden if isinstance(hidden, list) else [],
        "no_seed": no_seed if isinstance(no_seed, dict) else {},
    }


def hidden_species(workspace: Workspace) -> set[str]:
    """Species taken off the list in this folder; their files are kept."""
    return set(_removed_state(workspace)["hidden"])


def mark_removed(workspace: Workspace, species: str, *, hide: bool = False,
                 no_seed: tuple[str, ...] = ()) -> None:
    """Record that the user removed *species* from this folder.

    Seeding runs at GUI start and before every pipeline, so without this
    record deleted bundled data would simply come back. *hide* takes the
    species off the GUI's list; *no_seed* names the trees not to re-seed.
    """
    state = _removed_state(workspace)
    if hide and species not in state["hidden"]:
        state["hidden"].append(species)
    if no_seed:
        earlier = state["no_seed"].get(species, [])
        state["no_seed"][species] = sorted({*earlier, *no_seed})
    write_json(workspace.runtime_root / REMOVED_SPECIES_FILE, state)


def restore_species(workspace: Workspace, species: str) -> bool:
    """Forget that *species* was removed; True if there was a record of it."""
    state = _removed_state(workspace)
    if species not in state["hidden"] and species not in state["no_seed"]:
        return False
    state["hidden"] = [name for name in state["hidden"] if name != species]
    state["no_seed"].pop(species, None)
    write_json(workspace.runtime_root / REMOVED_SPECIES_FILE, state)
    return True


def _seed_skip(workspace: Workspace, tree: str) -> Callable[[Path], bool]:
    """Predicate on paths relative to *tree*: True for removed species' files."""
    no_seed = _removed_state(workspace)["no_seed"]
    removed = [name for name, trees in no_seed.items() if tree in trees]
    if tree == "data_json":
        return lambda relative: any(_species_data_file(name, relative) for name in removed)
    return lambda relative: relative.parts[0] in removed


def seed_bundled_data(workspace: Workspace, *, include_examples: bool = False) -> None:
    """Copy the bundled data into *workspace*; files the user or a stage changed are never touched.

    A copy that is still exactly as seeded is replaced when the bundled file changes (a new Qali
    version), so a folder never keeps an old input file. Each tree keeps a small record
    (``.qali_seeded.json``) of the copies it made. A file from before that record is adopted if it
    matches the bundle; an input file (literature data no stage writes) that does not is replaced,
    the old one kept in ``.qali_previous``; anything else is left alone.

    Pipeline inputs (``data_json``) are always copied. With
    *include_examples* the shipped example results (``plots``,
    ``orca_outputs``) are copied too. Trees the user removed for a species
    are skipped until :func:`restore_species`.
    """
    trees = dict(_INPUT_TREES)
    if include_examples:
        trees.update(_EXAMPLE_TREES)
    for tree, attribute in trees.items():
        _copy_missing(reference_path(tree), getattr(workspace, attribute),
                      _seed_skip(workspace, tree), is_input=_INPUT_FILE.search if tree == "data_json" else None)


def _stamp(path: Path) -> list[int]:
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def _same_bytes(a: Path, b: Path) -> bool:
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
        with a.open("rb") as fa, b.open("rb") as fb:
            while True:
                x, y = fa.read(1 << 20), fb.read(1 << 20)
                if x != y:
                    return False
                if not x:
                    return True
    except OSError:
        return False


def _copy_missing(source: Path, target: Path, skip: Callable[[Path], bool], is_input=None) -> None:
    """Copy every file under *source* that *target* lacks, except *skip* ones, and refresh untouched copies.

    Each file is copied under a temporary name and renamed into place, so an
    interrupted seed never leaves a truncated file that later runs would
    treat as present, and a concurrent reader sees a file absent or complete.
    """
    if not source.is_dir():
        return
    record_path = target / SEEDED_FILE
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            record = {}
    except (OSError, ValueError):
        record = {}
    changed = False
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if skip(relative):
            continue
        key = relative.as_posix()
        destination = target / relative
        if destination.exists():
            entry = record.get(key)
            try:
                mine, bundled = _stamp(destination), _stamp(path)
            except OSError:
                continue
            if entry is None:
                if _same_bytes(path, destination):           # a copy from before the record: adopt it
                    record[key] = {"bundle": bundled, "copy": mine}
                    changed = True
                    continue
                if not (is_input and is_input(key)) or not _keep_previous(destination, target / PREVIOUS_DIR / key):
                    continue                                 # results and unknown files stay as they are
            elif entry.get("copy") != mine:                  # edited, or rewritten by a stage: not ours now
                del record[key]
                changed = True
                continue
            elif entry.get("bundle") == bundled:
                continue                                     # bundle unchanged
        if _copy_into(path, destination):
            record[key] = {"bundle": _stamp(path), "copy": _stamp(destination)}
            changed = True
    if changed:
        _write_record(record_path, record)


def _keep_previous(destination: Path, aside: Path) -> bool:
    """Move an older input file out of the way before the bundled one replaces it."""
    try:
        aside.parent.mkdir(parents=True, exist_ok=True)
        os.replace(destination, aside)
        return True
    except OSError:
        return False


def _copy_into(path: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(handle)
    try:
        shutil.copy2(path, temp_name)
        os.replace(temp_name, destination)
        return True
    except OSError:
        Path(temp_name).unlink(missing_ok=True)
        # A concurrent seeder (e.g. a stage run) may have won the race.
        if not destination.exists():
            raise
        return False


def _write_record(record_path: Path, record: dict) -> None:
    try:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(prefix=".seeded.", suffix=".part", dir=record_path.parent)
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(record, fh)
        os.replace(temp_name, record_path)
    except OSError:
        pass                                                 # only a lost chance to refresh later
