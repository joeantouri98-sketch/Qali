"""The data a figure can plot: every table in a species' output files, with units and safe expressions.

A table is a list of records (``lifetimes``, ``transitions``, ``rates.states``, ...) or a set of numeric
arrays of equal length (the a(B) samples of a Feshbach run). Columns hold floats (``nan`` for a missing
value) or strings. Series, filters and derived quantities are expressions over the columns, evaluated
by :func:`evaluate`, which allows arithmetic, comparisons and numpy functions and nothing else.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import json
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

STAGES = ("rydberg", "transitions", "lifetimes", "polarizability", "blackbody", "tweezer", "hyperfine",
          "feshbach_out", "feshbach", "hf_constants")

# unit of a column from its name's suffix (mathtext, works with LaTeX too)
UNIT_SUFFIX = [
    ("_ns", "ns"), ("_us", r"$\mu$s"), ("_ms", "ms"), ("_s", "s"),
    ("_nm", "nm"), ("_um", r"$\mu$m"), ("_a0", r"$a_0$"), ("_au", "a.u."),
    ("_eV", "eV"), ("_meV", "meV"), ("_cm1", r"cm$^{-1}$"), ("_THz", "THz"),
    ("_GHz", "GHz"), ("_MHz", "MHz"), ("_kHz", "kHz"), ("_hz", "Hz"), ("_Hz", "Hz"),
    ("_G", "G"), ("_mT", "mT"), ("_K", "K"), ("_uK", r"$\mu$K"), ("_nK", "nK"),
    ("_muB", r"$\mu_B$"), ("_amu", "u"), ("_pct", "%"),
]
# quantities whose name is better shown as a symbol
NICE_NAMES = {
    "lifetime": r"$\tau$", "tau_rad": r"$\tau_\mathrm{rad}$", "tau_eff": r"$\tau_\mathrm{eff}$",
    "fosc": r"$f$", "n_star": r"$n^*$", "alpha0": r"$\alpha_0$", "alpha": r"$\alpha$",
    "wavelength": r"$\lambda$", "wl": r"$\lambda$", "energy": r"$E$", "delta_E": r"$\Delta E$",
    "A_total": r"$A$", "A": r"$A$", "Gamma_BBR": r"$\Gamma_\mathrm{BBR}$", "Gamma_rad": r"$\Gamma_\mathrm{rad}$",
    "B": r"$B$", "a_combined": r"$a$", "C6": r"$C_6$", "n": r"$n$", "l": r"$\ell$",
}


RUN_SEP = "__"                       # a run filed under a name: Na_lifetimes__hot.json (Save as)


@dataclass
class Table:
    key: str                         # "lifetimes:lifetimes", or "lifetimes__hot:lifetimes" for a named run
    species: str
    stage: str
    path: str                        # "lifetimes", "rates.states", "(arrays)"
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    run: str = ""                    # the Save as name; "" for the stage's usual output
    file: str = ""                   # a JSON file outside the output folder (plot builder)

    @property
    def rows(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0

    @property
    def title(self) -> str:
        if self.file:
            return f"{os.path.basename(self.file)}: {self.path or '(top)'} ({self.rows} rows)"
        stage = self.stage.replace('_', ' ').capitalize() + (f" ({self.run})" if self.run else "")
        return f"{stage}: {self.path} ({self.rows} rows)"

    def numeric_columns(self) -> list[str]:
        return [c for c, v in self.columns.items() if v.dtype.kind == "f"]


def unit_of(column: str) -> str:
    for suffix, unit in UNIT_SUFFIX:
        if column.endswith(suffix) and len(column) > len(suffix):
            return unit
    return ""


def label_for(column: str) -> str:
    """Axis label for a column: a symbol or readable name, then the unit in parentheses."""
    unit = unit_of(column)
    base = column
    for suffix, u in UNIT_SUFFIX:
        if u == unit and column.endswith(suffix):
            base = column[: -len(suffix)]
            break
    name = NICE_NAMES.get(base, base.replace("_", " "))
    return f"{name} ({unit})" if unit else name


# ---------------------------------------------------------------------------------------------- loading
def _as_column(values: list) -> np.ndarray:
    """Floats (None -> nan) when every value is a number or missing, else strings."""
    if all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) or isinstance(v, bool)
           for v in values):
        return np.array([math.nan if v is None else float(v) for v in values], dtype=float)
    return np.array(["" if v is None else str(v) for v in values], dtype=object)


def _records_table(records: list[dict]) -> dict[str, np.ndarray]:
    keys: list[str] = []
    for r in records:
        for k, v in r.items():
            if k not in keys and not isinstance(v, (dict, list)):
                keys.append(k)
    return {k: _as_column([r.get(k) for r in records]) for k in keys}


def _walk(node, path: str, out: list[tuple[str, dict]], depth: int = 0) -> None:
    if isinstance(node, dict):
        arrays = {k: v for k, v in node.items() if isinstance(v, list) and len(v) >= 2
                  and all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in v)}
        lengths = {len(v) for v in arrays.values()}
        if len(arrays) >= 2 and len(lengths) == 1:
            out.append((f"{path}(arrays)" if not path else f"{path} (arrays)",
                        {k: np.asarray(v, dtype=float) for k, v in arrays.items()}))
        if depth < 4:
            for k, v in node.items():
                _walk(v, f"{path}.{k}" if path else k, out, depth + 1)
    elif isinstance(node, list) and len(node) >= 2 and all(isinstance(e, dict) for e in node):
        cols = _records_table(node)
        if cols:
            out.append((path, cols))


def species_tables(species: str, data_dir: str) -> list[Table]:
    """Every table in the output files of *species* under *data_dir*, named runs included."""
    tables: list[Table] = []
    for stage in STAGES:
        for run in [""] + runs(species, stage, data_dir):
            doc = load_json(species, stage, data_dir, run=run)
            if doc is None:
                continue
            found: list[tuple[str, dict]] = []
            _walk(doc, "", found)
            tag = f"{stage}{RUN_SEP}{run}" if run else stage
            for sub, cols in found:
                tables.append(Table(f"{tag}:{sub}", species, stage, sub, cols, run))
    return tables


def find_table(species: str, data_dir: str, key: str) -> Table | None:
    return next((t for t in species_tables(species, data_dir) if t.key == key), None)


def file_tables(path: str) -> list[Table]:
    """Every table in any JSON file (records or equal-length arrays), keyed ``file:<path inside>``."""
    try:
        doc = _read(path)
    except (OSError, ValueError):
        return []
    found: list[tuple[str, dict]] = []
    _walk(doc, "", found)
    name = os.path.splitext(os.path.basename(path))[0]
    return [Table(f"file:{sub}", "", name, sub, cols, "", path) for sub, cols in found]


def find_file_table(path: str, key: str) -> Table | None:
    return next((t for t in file_tables(path) if t.key == key), None)


def runs(species: str, stage: str, data_dir: str) -> list[str]:
    """Names of the runs of *stage* for *species* filed with Save as (``Na_lifetimes__hot.json`` -> ``hot``)."""
    prefix = f"{species}_{stage}{RUN_SEP}"
    try:
        names = os.listdir(data_dir)
    except OSError:
        return []
    return sorted(n[len(prefix):-5] for n in names if n.startswith(prefix) and n.endswith(".json")
                  and len(n) > len(prefix) + 5)


def run_summary(species: str, stage: str, data_dir: str, run: str = "") -> str:
    """One line on what a run of *stage* is (the Stark and pair stages), "" when there is nothing to say."""
    if stage not in ("stark", "pair"):
        return ""
    doc = load_json(species, stage, data_dir, run)
    if not doc:
        return ""
    from qali.pipelines.rydberg_runs import summary
    return summary(stage, doc)


def stage_file(species: str, stage: str, data_dir: str, run: str = "") -> str:
    """The file a stage output is read from: the usual one, a named run, or (an absolute *run*) that file."""
    if run and os.path.isabs(run):
        return run
    return os.path.join(data_dir, f"{species}_{stage}{RUN_SEP + run if run else ''}.json")


# which run or file each (species, stage) is read from while a figure is drawn; see using_inputs
_INPUTS: contextvars.ContextVar[dict | None] = contextvars.ContextVar("figure_inputs", default=None)


@contextlib.contextmanager
def using_inputs(species: str, inputs: dict | None):
    """Within the block, :func:`load_json` reads *species*' stages from *inputs* ({stage: run name or path})."""
    token = _INPUTS.set({(species, st): v for st, v in (inputs or {}).items() if v})
    try:
        yield
    finally:
        _INPUTS.reset(token)


def available_species(data_dir: str) -> list[str]:
    out = set()
    if os.path.isdir(data_dir):
        for name in os.listdir(data_dir):
            m = re.match(r"^([A-Z][a-z]?(?:_c\d+)?)_(lifetimes|rydberg|polarizability)(?:__.+)?\.json$", name)
            if m:
                out.add(m.group(1))
    return sorted(out)


_CACHE: dict[str, tuple[float, int, object]] = {}


def _read(path: str):
    """Parsed JSON of *path*, read again only when the file changed (size or time)."""
    st = os.stat(path)
    hit = _CACHE.get(path)
    if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2]
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if len(_CACHE) > 64:
        _CACHE.clear()
    _CACHE[path] = (st.st_mtime, st.st_size, doc)
    return doc


def load_json(species: str, stage: str, data_dir: str, run: str | None = None) -> dict | None:
    """The stage's output for *species*, or None. *run* picks a named run or a file; by default the one
    chosen with :func:`using_inputs`, else the usual output. Treat the result as read-only (it is cached)."""
    if run is None:
        run = (_INPUTS.get() or {}).get((species, stage), "")
    try:
        doc = _read(stage_file(species, stage, data_dir, run))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


# ------------------------------------------------------------------------------------------ expressions
_FUNCS = {
    "log": np.log, "log10": np.log10, "log2": np.log2, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "arcsin": np.arcsin, "arccos": np.arccos,
    "arctan": np.arctan, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "floor": np.floor,
    "ceil": np.ceil, "round": np.round, "minimum": np.minimum, "maximum": np.maximum,
    "isnan": np.isnan, "isfinite": np.isfinite, "where": np.where,
}
_CONSTS = {"pi": math.pi, "e": math.e, "nan": math.nan, "inf": math.inf}
_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
           ast.Pow: np.power, ast.Mod: np.mod, ast.FloorDiv: np.floor_divide}
_CMPOPS = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
           ast.Eq: np.equal, ast.NotEq: np.not_equal}


class ExpressionError(ValueError):
    pass


def evaluate(expr: str, columns: dict[str, np.ndarray]):
    """Value of *expr* over *columns*: column names, numbers, strings, + - * / ** % //, comparisons,
    and / or / not and & | ~ (element-wise), the functions in ``_FUNCS`` and ``col("name")`` for a column whose
    name is not an identifier. Anything else (attributes, indexing, imports) is refused."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Cannot read '{expr}': {exc.msg}.") from None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in columns:
                return columns[node.id]
            if node.id in _CONSTS:
                return _CONSTS[node.id]
            raise ExpressionError(f"No column '{node.id}'.")
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            fn = np.logical_and if isinstance(node.op, ast.BitAnd) else np.logical_or
            return fn(ev(node.left), ev(node.right))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            with np.errstate(all="ignore"):
                return _BINOPS[type(node.op)](_num(ev(node.left)), _num(ev(node.right)))
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            if isinstance(node.op, ast.USub):
                return np.negative(_num(v))
            if isinstance(node.op, ast.UAdd):
                return _num(v)
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return np.logical_not(v)
        if isinstance(node, ast.BoolOp):
            vals = [ev(v) for v in node.values]
            fn = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            out = vals[0]
            for v in vals[1:]:
                out = fn(out, v)
            return out
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            result = None
            for op, comp in zip(node.ops, node.comparators):
                if type(op) not in _CMPOPS:
                    raise ExpressionError("Only <, <=, >, >=, == and != compare values.")
                right = ev(comp)
                with np.errstate(invalid="ignore"):
                    part = _CMPOPS[type(op)](left, right)
                result = part if result is None else np.logical_and(result, part)
                left = right
            return result
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            if node.func.id == "col" and len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                name = str(node.args[0].value)
                if name not in columns:
                    raise ExpressionError(f"No column '{name}'.")
                return columns[name]
            fn = _FUNCS.get(node.func.id)
            if fn is None:
                raise ExpressionError(f"Unknown function '{node.func.id}'.")
            with np.errstate(all="ignore"):
                return fn(*[_num(ev(a)) if node.func.id != "where" or i else ev(a)
                            for i, a in enumerate(node.args)])
        raise ExpressionError(f"'{expr}' uses something an expression may not contain.")

    return ev(tree)


def _num(v):
    if isinstance(v, np.ndarray) and v.dtype == object:
        raise ExpressionError("A text column cannot be used in arithmetic.")
    if isinstance(v, str):
        raise ExpressionError("Text cannot be used in arithmetic.")
    return v


def evaluate_column(expr: str, table: Table) -> np.ndarray:
    """*expr* over *table* as an array of the table's length (a constant is broadcast)."""
    v = evaluate(expr, table.columns)
    arr = np.asarray(v)
    if arr.ndim == 0:
        arr = np.full(table.rows, arr.item())
    if arr.shape != (table.rows,):
        raise ExpressionError(f"'{expr}' does not give one value per row.")
    return arr
