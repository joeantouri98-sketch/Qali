"""Fine-structure-resolved Rydberg energies of the alkali atoms: quantum defects δ(n, l, j).

Energies are E(n, l, j) = E_I - R_M / (n - δ)², with the modified Ritz expansion

    δ(n, l, j) = δ0 + δ2/(n - δ0)² + δ4/(n - δ0)⁴ + δ6/(n - δ0)⁶,

R_M the Rydberg constant of the isotope (reduced mass of the electron and the singly charged core)
and E_I the ionization energy the defects were fitted with (they belong together).

Where each level comes from:
  1. n below the table's ``n_min`` (the low levels the precision fits leave out), and any level of
     a series the table lacks: the NIST level;
  2. a series in the table: the Ritz formula above (precision microwave and laser spectroscopy);
  3. Fr, and any series NIST has but the table lacks: δ0 and δ2 fitted to the NIST levels of that
     (l, j), the lowest one left out when there are three or more;
  4. higher l: the core-polarisation defect δ = α_d (3n² - l(l+1)) / (4n² (l-½) l (l+½)(l+1)(l+3/2)),
     the first-order shift -α_d/2 <r⁻⁴> of a hydrogenic orbital (Rb g: 0.0039 vs 0.0040 measured).

Sources (isotope; the values ARC uses, Šibalić et al., CPC 220, 319 (2017)):
  Li-7   s, p: Goy et al., PRA 34, 2889 (1986); d, f: Lorenzen & Niemax, Phys. Scr. 27, 300 (1983)
  Na-23  Lorenzen & Niemax, Phys. Scr. 27, 300 (1983)
  K-39   Peper et al., PRA 100, 012501 (2019), with their E_I and R_K
  Rb-85  s, p, d: Li et al., PRA 67, 052502 (2003); f: Han et al., PRA 74, 054502 (2006);
         g: Moore et al., PRA 102, 062817 (2020); E_I: Sanguinetti et al., J. Phys. B 42, 165004 (2009)
  Cs-133 s, p, d5/2: Deiglmayr et al., PRA 93, 013424 (2016), with their E_I; d3/2: Lorenzen &
         Niemax, Z. Phys. A 315, 127 (1984); f, g: Weber & Sansonetti, PRA 35, 4650 (1987)
Editable in Preferences > Tables and variables.

Engine contract: inputs as arguments, no files read here (the caller passes the NIST levels).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from qali.core import user_tables as _user_tables

R_INF_CM = 109737.31568160          # Rydberg constant, cm-1 (CODATA 2018)
M_E_AMU = 5.48579909065e-4          # electron mass, u
L_LETTERS = "spdfghiklmnoqrtuvwxyz"

# {species: {mass_amu, ionization_cm, n_min, "<l><j>": {d0, d2, d4, d6}}}; f and g have no j
# dependence in the data and repeat the same numbers for both j.
RYDBERG_DEFECTS: dict[str, dict] = {
    "Li": {"mass_amu": 7.0160034366, "ionization_cm": 43487.1465, "n_min": 4,
           "s1/2": {"d0": 0.3995101, "d2": 0.0290},
           "p1/2": {"d0": 0.0471780, "d2": -0.024},
           "p3/2": {"d0": 0.0471665, "d2": -0.024},
           "d3/2": {"d0": 0.002129, "d2": -0.01491, "d4": 0.1759, "d6": -0.8507},
           "d5/2": {"d0": 0.002129, "d2": -0.01491, "d4": 0.1759, "d6": -0.8507},
           "f5/2": {"d0": -0.000077, "d2": 0.021856, "d4": -0.4211, "d6": 2.3891},
           "f7/2": {"d0": -0.000077, "d2": 0.021856, "d4": -0.4211, "d6": 2.3891}},
    "Na": {"mass_amu": 22.9897692820, "ionization_cm": 41449.44, "n_min": 5,
           "s1/2": {"d0": 1.347964, "d2": 0.060673, "d4": 0.0233, "d6": -0.0085},
           "p1/2": {"d0": 0.855380, "d2": 0.11363, "d4": 0.0384, "d6": 0.1412},
           "p3/2": {"d0": 0.854565, "d2": 0.114195, "d4": 0.0352, "d6": 0.1533},
           "d3/2": {"d0": 0.015543, "d2": -0.08535, "d4": 0.7958, "d6": -4.0513},
           "d5/2": {"d0": 0.015543, "d2": -0.08535, "d4": 0.7958, "d6": -4.0513},
           "f5/2": {"d0": 0.001453, "d2": 0.017312, "d4": -0.7809, "d6": 7.021},
           "f7/2": {"d0": 0.001453, "d2": 0.017312, "d4": -0.7809, "d6": 7.021}},
    "K": {"mass_amu": 38.9637064864, "ionization_cm": 35009.8139710, "n_min": 6,
          "s1/2": {"d0": 2.18020826, "d2": 0.134534, "d4": 0.0952, "d6": 0.0021},
          "p1/2": {"d0": 1.71392626, "d2": 0.23114, "d4": 0.1948, "d6": 0.3683},
          "p3/2": {"d0": 1.71087854, "d2": 0.23233, "d4": 0.1961, "d6": 0.3716},
          "d3/2": {"d0": 0.27698453, "d2": -1.02691, "d4": -0.665, "d6": 10.9},
          "d5/2": {"d0": 0.27715665, "d2": -1.02493, "d4": -0.640, "d6": 10.0},
          "f5/2": {"d0": 0.0094576, "d2": -0.0446},
          "f7/2": {"d0": 0.0094576, "d2": -0.0446},
          "g7/2": {"d0": 0.0024080, "d2": -0.0209},
          "g9/2": {"d0": 0.0024080, "d2": -0.0209}},
    "Rb": {"mass_amu": 84.9117897379, "ionization_cm": 33690.7975183, "n_min": 10,
           "s1/2": {"d0": 3.1311804, "d2": 0.1784},
           "p1/2": {"d0": 2.6548849, "d2": 0.2900},
           "p3/2": {"d0": 2.6416737, "d2": 0.2950},
           "d3/2": {"d0": 1.34809171, "d2": -0.60286},
           "d5/2": {"d0": 1.34646572, "d2": -0.59600},
           "f5/2": {"d0": 0.0165192, "d2": -0.085},
           "f7/2": {"d0": 0.0165437, "d2": -0.086},
           "g7/2": {"d0": 0.0039990, "d2": -0.0202},
           "g9/2": {"d0": 0.0039990, "d2": -0.0202}},
    "Cs": {"mass_amu": 132.9054519610, "ionization_cm": 31406.4677325, "n_min": 12,
           "s1/2": {"d0": 4.0493532, "d2": 0.2391, "d4": 0.06, "d6": 11.0, "d8": -209.0},
           "p1/2": {"d0": 3.5915871, "d2": 0.36273},
           "p3/2": {"d0": 3.5590676, "d2": 0.37469},
           "d3/2": {"d0": 2.4754562, "d2": 0.009320, "d4": -0.43498, "d6": -0.76358, "d8": -18.0061},
           "d5/2": {"d0": 2.4663144, "d2": 0.01381, "d4": -0.392, "d6": -1.9},
           "f5/2": {"d0": 0.03341424, "d2": -0.198674, "d4": 0.28953, "d6": -0.2601},
           "f7/2": {"d0": 0.03341424, "d2": -0.198674, "d4": 0.28953, "d6": -0.2601},
           "g7/2": {"d0": 0.00703865, "d2": -0.049252, "d4": 0.01291},
           "g9/2": {"d0": 0.00703865, "d2": -0.049252, "d4": 0.01291}},
}
_user_tables.apply("rydberg.defects", RYDBERG_DEFECTS)

# Fr has no precision Rydberg data: NIST levels and the fit to them (rule 3), and this mass.
FR_MASS_AMU = 223.0197360


def l_letter(l: int) -> str:
    return L_LETTERS[l] if l < len(L_LETTERS) else f"[l={l}]"


def j_label(l: int, j2: int) -> str:
    return f"{l_letter(l)}{j2}/2"


def rydberg_cm(mass_amu: float) -> float:
    """R_M for a neutral atom of this mass: the electron against the singly charged core."""
    m_core = mass_amu - M_E_AMU
    return R_INF_CM * m_core / (m_core + M_E_AMU)


def ritz(coef: Mapping[str, float], n: int) -> float:
    d0 = coef.get("d0", 0.0)
    x = 1.0 / (n - d0) ** 2
    return d0 + sum(coef.get(f"d{2 * k}", 0.0) * x ** k for k in range(1, 5))


def polarization_defect(alpha_d: float, n: int, l: int) -> float:
    """First-order core-polarisation defect of a hydrogenic n, l orbital (l >= 1)."""
    den = (l - 0.5) * l * (l + 0.5) * (l + 1) * (l + 1.5)
    return alpha_d * (3 * n * n - l * (l + 1)) / (4 * n * n * den)


@dataclass(frozen=True)
class Level:
    n: int
    l: int
    j2: int
    binding_cm: float          # E_I - E (positive)
    n_star: float              # effective quantum number on R_M
    source: str                # 'NIST' | 'table' | 'NIST fit' | 'polarisation'

    @property
    def label(self) -> str:
        return f"{self.n}{j_label(self.l, self.j2)}"


class RydbergEnergies:
    """Level energies of one alkali species, fine structure resolved.

    nist          {(n, l letter, J string): excitation energy, cm-1} (NIST export of the species)
    nist_ip_cm    the NIST ionization limit, cm-1 (used for Fr and to place NIST levels when the
                  table has no E_I of its own)
    alpha_d       core dipole polarisability, a.u. (for the high-l defects)
    """

    def __init__(self, species: str, nist: Mapping[tuple[int, str, str], float], nist_ip_cm: float | None,
                 alpha_d: float, mass_amu: float | None = None):
        self.species = species
        tab = RYDBERG_DEFECTS.get(species)
        self.table = tab or {}
        self.alpha_d = alpha_d
        if tab:
            self.ip_cm = float(tab["ionization_cm"])
            self.rm_cm = rydberg_cm(float(tab["mass_amu"]))
            self.n_min = int(tab.get("n_min", 0))
        else:
            if nist_ip_cm is None:
                raise ValueError(f"No Rydberg table for {species} and no NIST ionization limit.")
            self.ip_cm = float(nist_ip_cm)
            self.rm_cm = rydberg_cm(mass_amu or FR_MASS_AMU)
            self.n_min = 10 ** 6                       # NIST levels wherever they exist
        self.nist: dict[tuple[int, int, int], float] = {}
        for (n, lc, J), e in nist.items():
            jf = _jfloat(J)
            if lc in L_LETTERS[:8] and jf is not None:
                self.nist[(n, L_LETTERS.index(lc), int(round(2 * jf)))] = float(e)
        self.ground_n = min((n for (n, l, _j) in self.nist if l == 0), default=0)
        self._fits: dict[tuple[int, int], dict[str, float]] = {}
        self._fit_nist()

    # ---- fits to NIST for series the table lacks
    def _fit_nist(self) -> None:
        series: dict[tuple[int, int], list[tuple[int, float]]] = {}
        for (n, l, j2), e in self.nist.items():
            series.setdefault((l, j2), []).append((n, e))
        for (l, j2), pts in series.items():
            if j_label(l, j2) in self.table:
                continue
            pts.sort()
            if len(pts) >= 3:
                pts = pts[1:]                           # the lowest member is the most perturbed
            ns = np.array([n for n, _ in pts], float)
            nst = np.sqrt(self.rm_cm / (self.ip_cm - np.array([e for _, e in pts])))
            dl = ns - nst
            if len(pts) == 1:
                self._fits[(l, j2)] = {"d0": float(dl[0])}
                continue
            d0 = float(dl[-1])
            for _ in range(20):                         # δ = d0 + d2/(n-d0)²: linear in (d0, d2) at fixed n-d0
                A = np.column_stack([np.ones_like(ns), 1.0 / (ns - d0) ** 2])
                sol, *_ = np.linalg.lstsq(A, dl, rcond=None)
                if abs(sol[0] - d0) < 1e-12:
                    break
                d0 = float(sol[0])
            self._fits[(l, j2)] = {"d0": float(sol[0]), "d2": float(sol[1])}

    def defect(self, n: int, l: int, j2: int) -> tuple[float, str]:
        key = j_label(l, j2)
        if key in self.table and n >= self.n_min:
            return ritz(self.table[key], n), "table"
        if (l, j2) in self._fits:
            return ritz(self._fits[(l, j2)], n), "NIST fit"
        if key in self.table:
            return ritz(self.table[key], n), "table"
        if l == 0:
            raise ValueError(f"No s-series data for {self.species}.")
        return polarization_defect(self.alpha_d, n, l), "polarisation"

    def level(self, n: int, l: int, j2: int) -> Level:
        if n < self.n_min and (n, l, j2) in self.nist:
            b = self.ip_cm - self.nist[(n, l, j2)]
            return Level(n, l, j2, b, math.sqrt(self.rm_cm / b), "NIST")
        if (n, l, j2) in self.nist and (not self.table or j_label(l, j2) not in self.table):
            b = self.ip_cm - self.nist[(n, l, j2)]
            return Level(n, l, j2, b, math.sqrt(self.rm_cm / b), "NIST")
        d, src = self.defect(n, l, j2)
        ns = n - d
        return Level(n, l, j2, self.rm_cm / ns ** 2, ns, src)

    def exists(self, n: int, l: int) -> bool:
        """True for the physical levels: l < n, and not inside the core (below the ground n for the
        penetrating series unless NIST lists the level, e.g. Rb 4d, 4f)."""
        if l >= n:
            return False
        if n >= self.ground_n:
            return True
        return any((n, l, j2) in self.nist for j2 in (2 * l - 1, 2 * l + 1))


def _jfloat(J: str) -> float | None:
    try:
        if "/" in J:
            a, b = J.split("/")
            return float(a) / float(b)
        return float(J)
    except (ValueError, ZeroDivisionError):
        return None
