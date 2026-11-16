"""How the app shows numbers: the units and number format chosen in Edit > Preferences > Units and numbers.

Display only. The stages, their JSON files, the HTML pages and exported CSV keep their own units
(eV, vacuum nm, s, Hz, atomic units, a0); this module converts at the moment a value is shown.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace

# 2022 CODATA
EV_TO_CM = 8065.543937
EV_TO_THZ = 241.7989242
EV_TO_HARTREE = 1 / 27.211386245988
AU_TO_HZ_VCM2 = 0.2488319          # polarizability: 1 a.u. in h·Hz/(V/cm)²
AU_TO_A3 = 0.1481847114            # polarizability: 1 a.u. = a0³ in Å³
A0_TO_NM = 0.0529177210544

ENERGY_UNITS = {"eV": "eV", "meV": "meV", "cm-1": "cm⁻¹", "THz": "THz", "Eh": "Eₕ"}
POLARIZABILITY_UNITS = {"au": "a.u.", "hHz": "h·Hz/(V/cm)²", "A3": "Å³"}
LENGTH_UNITS = {"a0": "a₀", "nm": "nm"}
WAVELENGTH_MODES = {"vacuum": "Vacuum", "air": "Air"}
SCALE_MODES = {"auto": "Automatic prefix", "base": "Always the base unit"}

KEY = "units/{name}"


@dataclass(frozen=True)
class Units:
    energy: str = "eV"
    wavelength: str = "vacuum"
    time: str = "auto"
    frequency: str = "auto"
    polarizability: str = "au"
    length: str = "a0"
    sig_figs: int = 6
    sci_above: float = 1e5
    sci_below: float = 1e-4


DEFAULTS = Units()
_current = DEFAULTS


def current() -> Units:
    return _current


def set_current(units: Units) -> None:
    global _current
    _current = units


def load(settings) -> Units:
    """The units saved in *settings* (defaults for anything missing or invalid)."""
    values = {}
    for f in fields(Units):
        raw = settings.value(KEY.format(name=f.name), None)
        if raw is None:
            continue
        try:
            values[f.name] = type(getattr(DEFAULTS, f.name))(raw)
        except (TypeError, ValueError):
            continue
    units = replace(DEFAULTS, **values)
    if units.energy not in ENERGY_UNITS or units.polarizability not in POLARIZABILITY_UNITS \
            or units.length not in LENGTH_UNITS or units.wavelength not in WAVELENGTH_MODES:
        units = DEFAULTS
    set_current(units)
    return units


def save(settings, units: Units) -> None:
    """Save *units*; values equal to the defaults are removed rather than written."""
    for name, value in asdict(units).items():
        key = KEY.format(name=name)
        if value == getattr(DEFAULTS, name):
            settings.remove(key)
        else:
            settings.setValue(key, value)


def changed(units: Units) -> int:
    return sum(getattr(units, f.name) != getattr(DEFAULTS, f.name) for f in fields(Units))


# ---- numbers ------------------------------------------------------------------------------------------
def number(x: float, u: Units | None = None) -> str:
    """*x* with the chosen significant figures, in scientific notation outside the chosen range."""
    u = u or _current
    if x == 0:
        return "0"
    if not math.isfinite(x):
        return str(x)
    if abs(x) >= u.sci_above or abs(x) < u.sci_below:
        return f"{x:.{max(u.sig_figs - 1, 0)}e}"
    return f"{x:.{u.sig_figs}g}"


def vacuum_to_air_nm(lam_nm: float) -> float:
    """Air wavelength (standard air, Edlén 1966 as in NIST ASD) for a vacuum wavelength above 200 nm."""
    if lam_nm < 200:
        return lam_nm
    s2 = (1e3 / lam_nm) ** 2                     # (1/λ in µm)²
    n = 1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2)
    return lam_nm / n


_PREFIXES = ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"),
             (1e-9, "n"), (1e-12, "p"), (1e-15, "f"))


def _prefixed(x: float, base: str, allowed: str) -> tuple[float, str]:
    for factor, p in _PREFIXES:
        if p in allowed and abs(x) >= factor:
            return x / factor, p + base
    factor, p = next((f, q) for f, q in reversed(_PREFIXES) if q in allowed)
    return x / factor, p + base


def convert(value: float, kind: str, u: Units | None = None) -> tuple[float, str]:
    """(value, unit) for display. *value* is in the stored unit of *kind*:
    energy eV, wavelength vacuum nm, time s, frequency Hz, polarizability a.u., length a0."""
    u = u or _current
    if kind == "energy":
        factor = {"eV": 1.0, "meV": 1e3, "cm-1": EV_TO_CM, "THz": EV_TO_THZ, "Eh": EV_TO_HARTREE}[u.energy]
        return value * factor, ENERGY_UNITS[u.energy]
    if kind == "wavelength":
        if u.wavelength == "air" and value >= 200:
            return vacuum_to_air_nm(value), "nm (air)"
        return value, "nm"
    if kind == "time":
        if u.time == "base" or value == 0:
            return value, "s"
        return _prefixed(value, "s", " mµnp")
    if kind == "frequency":
        if u.frequency == "base" or value == 0:
            return value, "Hz"
        return _prefixed(value, "Hz", "TGMk ")
    if kind == "polarizability":
        factor = {"au": 1.0, "hHz": AU_TO_HZ_VCM2, "A3": AU_TO_A3}[u.polarizability]
        return value * factor, POLARIZABILITY_UNITS[u.polarizability]
    if kind == "length":
        return (value * A0_TO_NM, "nm") if u.length == "nm" else (value, "a₀")
    raise ValueError(f"unknown kind {kind!r}")


def show(value, kind: str, u: Units | None = None) -> str:
    """'value unit' for display; non-numbers as they are."""
    if value is None:
        return "—"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    shown, unit = convert(x, kind, u)
    return f"{number(shown, u)} {unit}"
