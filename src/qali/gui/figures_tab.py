"""The Figures tab: publication figures drawn with Matplotlib (templates and the plot builder).

The left side edits a figure recipe (template and its data, axes, size and style); the right side shows the
figure at its printed proportions. Export writes PDF, SVG, EPS, PNG or TIFF; Script writes a standalone
Python file that draws the same figure; recipes are saved as ``.qfig`` and reopened later.
"""

from __future__ import annotations

import copy
import csv
import os

from PySide6.QtCore import QEvent, QPoint, QRect, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QRubberBand,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qali.figures import catalog, templates
from qali.figures import recipe as R
from qali.figures import spec as S
from qali.figures.style import FONTS, JOURNALS, PALETTES, latex_problem

from .plots_page import load_defaults, save_defaults

OTHER_FILE = "\x00other"                  # the "Other file..." entry of an input list
LAST_DIR_KEY = "figures/last_dir"
EXPORT_FILTERS = ";;".join(f"{label} (*.{ext})" for ext, label in S.SAVE_FORMATS.items())


def _settings() -> QSettings:
    from .main_window import open_settings

    return open_settings()


def _int_or_none(text: str):
    v = _float_or_none(text)
    return int(round(v)) if v is not None else None


def _float_or_none(text: str):
    text = text.strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


MARKER_CHOICES = (("automatic", ""), ("circle", "o"), ("square", "s"), ("triangle", "^"), ("diamond", "D"),
                  ("down triangle", "v"), ("left triangle", "<"), ("right triangle", ">"),
                  ("pentagon", "p"), ("hexagon", "h"), ("star", "*"), ("plus", "P"), ("cross", "X"),
                  ("thin plus", "+"), ("thin cross", "x"), ("point", "."))

class FiguresTab(QWidget):
    """One figure editor; the window has one in the viewer tabs and one in the bottom panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_dir = ""
        self.species = ""
        self.recipe = R.new_recipe("grotrian")
        self._apply_saved_defaults()
        self.spec: S.FigureSpec | None = None
        self._stale = True
        self._building = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.redraw)
        self._build()
        self._load_controls()

    # ------------------------------------------------------------------ context
    def set_context(self, data_dir: str, species: str) -> None:
        """The output folder's data directory and the selected species; redraws when showing."""
        self.data_dir = data_dir
        if species:
            self.species = species
        self._fill_species()
        self._fill_inputs()                # runs saved under a name may have come or gone
        self._fill_choices()
        self._stale = True                 # a stage may have rewritten the files
        if self.isVisible():
            self._schedule()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._stale:
            self._schedule()

    def _apply_saved_defaults(self) -> None:
        d = load_defaults(_settings())
        self.recipe["style"].update(d["style"])
        self.recipe["size"].update(d["size"])

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)

        side = QWidget()
        sv = QVBoxLayout(side)
        sv.setContentsMargins(8, 8, 8, 8)
        top = QFormLayout()
        self.template = QComboBox()
        for key, (label, _fn) in templates.TEMPLATES.items():
            self.template.addItem(label, key)
        self.template.setToolTip("What to draw")
        top.addRow("Figure", self.template)
        self.species_box = QComboBox()
        self.species_box.setToolTip("Species whose output files are drawn")
        top.addRow("Species", self.species_box)
        sv.addLayout(top)
        row = QHBoxLayout()
        for text, tip, slot in (("Open...", "Open a saved figure recipe (.qfig)", self._open),
                                ("Save...", "Save this figure's recipe (.qfig)", self._save_recipe),
                                ("Export...", "Save the figure: PDF, SVG, EPS, PNG or TIFF", self._export),
                                ("Script...", "Write a Python script that draws this figure", self._script),
                                ("Data...", "Save the plotted numbers as CSV", self._csv)):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        sv.addLayout(row)

        body = QWidget()
        bv = QVBoxLayout(body)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.addWidget(self._inputs_group())
        self.params_box = QGroupBox("Data")
        pv = QVBoxLayout(self.params_box)
        self.param_pages: dict[str, QWidget] = {}
        for key in templates.TEMPLATES:              # one page per template; only the current one is shown
            page = getattr(self, f"_page_{key}")()
            self.param_pages[key] = page
            page.setVisible(False)
            pv.addWidget(page)
        bv.addWidget(self.params_box)
        bv.addWidget(self._axes_group())
        bv.addWidget(self._figure_group())
        bv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(330)
        self.controls_scroll = scroll
        body.installEventFilter(self)            # keep the panel at least as wide as its widest row
        sv.addWidget(scroll, 1)
        split.addWidget(side)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 6, 6, 6)
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(3.375, 2.5))
        self.canvas_holder = QWidget()
        self.canvas_holder.setObjectName("figureHolder")
        hl = QVBoxLayout(self.canvas_holder)
        self.preview = QLabel()                    # the figure as an image at the printed proportions
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(200, 150)
        self.preview.installEventFilter(self)
        self._pan: tuple | None = None
        self._box_start: QPoint | None = None
        self._band = QRubberBand(QRubberBand.Rectangle, self.preview)
        self._live = QTimer(self)
        self._live.setSingleShot(True)
        self._live.setInterval(30)
        self._live.timeout.connect(self._render_preview)
        self.view_zoom = 1.0                       # whole-figure magnification (Ctrl + wheel); 1 = fit the panel
        self._scroll_drag: QPoint | None = None
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidget(self.preview)
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setAlignment(Qt.AlignCenter)
        self.preview_scroll.setFrameShape(QScrollArea.NoFrame)
        hl.addWidget(self.preview_scroll, 1)
        rv.addWidget(self.canvas_holder, 1)
        from .popout import add_zoom_shortcuts

        add_zoom_shortcuts(self.canvas_holder, self._view_zoom_step)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(120)
        self._resize_timer.timeout.connect(self._render_preview)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rv.addWidget(self.message)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 700])

        self.template.currentIndexChanged.connect(self._template_changed)
        self.species_box.currentIndexChanged.connect(self._species_changed)

    # ----- axes and figure groups
    def _axes_group(self) -> QGroupBox:
        g = QGroupBox("Axes")
        f = QFormLayout(g)
        self.xlabel, self.ylabel, self.title_edit = QLineEdit(), QLineEdit(), QLineEdit()
        for w, tip in ((self.xlabel, "x-axis label; $...$ for maths, e.g. $n^*$"),
                       (self.ylabel, "y-axis label; $...$ for maths, e.g. $\\tau$ (ns)"),
                       (self.title_edit, "Usually empty in papers: the caption carries the title")):
            w.setPlaceholderText("automatic")
            w.setToolTip(tip)
            w.editingFinished.connect(self._controls_changed)
        f.addRow("x label", self.xlabel)
        f.addRow("y label", self.ylabel)
        f.addRow("Title", self.title_edit)
        self.xscale, self.yscale = QComboBox(), QComboBox()
        for box in (self.xscale, self.yscale):
            box.addItem("automatic", "")
            for s in R.SCALES:
                box.addItem(s, s)
            box.currentIndexChanged.connect(self._controls_changed)
        f.addRow("x scale", self.xscale)
        f.addRow("y scale", self.yscale)
        self.xmin, self.xmax, self.ymin, self.ymax = (QLineEdit() for _ in range(4))
        for w in (self.xmin, self.xmax, self.ymin, self.ymax):
            w.setPlaceholderText("auto")
            w.editingFinished.connect(self._controls_changed)
        f.addRow("x range", self._pair(self.xmin, self.xmax))
        f.addRow("y range", self._pair(self.ymin, self.ymax))
        self.legend = QComboBox()
        for text, key in (("automatic", "auto"), ("shown", "on"), ("hidden", "off")):
            self.legend.addItem(text, key)
        self.legend_loc = QComboBox()
        self.legend_loc.addItems(R.LEGEND_LOCS)
        self.legend_ncol = QSpinBox()
        self.legend_ncol.setRange(1, 6)
        self.legend_frame = QCheckBox("Frame")
        for w in (self.legend, self.legend_loc):
            w.currentIndexChanged.connect(self._controls_changed)
        self.legend_ncol.valueChanged.connect(self._controls_changed)
        self.legend_frame.toggled.connect(self._controls_changed)
        f.addRow("Legend", self.legend)
        f.addRow("Position", self.legend_loc)
        f.addRow("Columns", self._pair(self.legend_ncol, self.legend_frame))
        return g

    def _figure_group(self) -> QGroupBox:
        g = QGroupBox("Size and style")
        f = QFormLayout(g)
        self.journal = QComboBox()
        for key, j in JOURNALS.items():
            self.journal.addItem(j.label, key)
        self.journal.setToolTip("Sets the printed width and the font size")
        self.columns = QComboBox()
        self.columns.addItem("One column", 1)
        self.columns.addItem("Full width", 2)
        self.aspect = QDoubleSpinBox()
        self.aspect.setRange(0.0, 3.0)
        self.aspect.setSingleStep(0.05)
        self.aspect.setSpecialValueText("template")
        self.aspect.setToolTip("Height / width; 'template' uses the figure's own")
        self.width_cm, self.height_cm = QDoubleSpinBox(), QDoubleSpinBox()
        for w in (self.width_cm, self.height_cm):
            w.setRange(0.0, 40.0)
            w.setSingleStep(0.5)
            w.setSuffix(" cm")
            w.setSpecialValueText("preset")
        self.palette = QComboBox()
        for key, (label, _cols) in PALETTES.items():
            self.palette.addItem(label, key)
        self.font_box = QComboBox()
        for key, label in FONTS.items():
            self.font_box.addItem(label, key)
        self.font_size = QDoubleSpinBox()
        self.font_size.setRange(5.0, 24.0)
        self.font_size.setSingleStep(0.5)
        self.font_size.setSuffix(" pt")
        self.usetex = QCheckBox("Typeset text with LaTeX")
        problem = latex_problem(raster=False)
        self.usetex.setToolTip(problem or "Slower; needs LaTeX with the cm-super fonts")
        self.ticks_in = QCheckBox("Ticks inside")
        self.mirror = QCheckBox("Ticks on all sides")
        self.minor = QCheckBox("Minor ticks")
        self.grid = QCheckBox("Grid")
        self.panel_labels = QCheckBox("Panel labels (a), (b)")
        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(0.3, 4.0)
        self.line_width.setSingleStep(0.1)
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(1.0, 12.0)
        self.marker_size.setSingleStep(0.5)
        for w in (self.journal, self.columns, self.palette, self.font_box):
            w.currentIndexChanged.connect(self._controls_changed)
        for w in (self.aspect, self.width_cm, self.height_cm, self.font_size, self.line_width, self.marker_size):
            w.valueChanged.connect(self._controls_changed)
        for w in (self.usetex, self.ticks_in, self.mirror, self.minor, self.grid, self.panel_labels):
            w.toggled.connect(self._controls_changed)
        f.addRow("Preset", self.journal)
        f.addRow("Width", self.columns)
        f.addRow("Aspect", self.aspect)
        f.addRow("Custom", self._pair(self.width_cm, self.height_cm))
        f.addRow("Colours", self.palette)
        f.addRow("Font", self.font_box)
        f.addRow("Font size", self.font_size)
        f.addRow(self.usetex)
        f.addRow(self._pair(self.ticks_in, self.mirror))
        f.addRow(self._pair(self.minor, self.grid))
        f.addRow(self.panel_labels)
        f.addRow("Line width", self.line_width)
        f.addRow("Marker size", self.marker_size)
        b = QPushButton("Use as default")
        b.setToolTip("New figures start with this size and style")
        b.clicked.connect(self._save_defaults)
        f.addRow(b)
        return g

    @staticmethod
    def _pair(a, b) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(b)
        return w

    # ----- template pages
    def _checks(self, items, tip="") -> tuple[QWidget, dict[str, QCheckBox]]:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        boxes = {}
        for key, text in items:
            c = QCheckBox(text)
            c.setToolTip(tip)
            c.toggled.connect(self._controls_changed)
            h.addWidget(c)
            boxes[key] = c
        h.addStretch(1)
        return w, boxes

    def _page_builder(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        self.series_editor = SeriesEditor(self)
        self.series_editor.changed.connect(self._controls_changed)
        v.addWidget(self.series_editor)
        f = QFormLayout()
        f.setContentsMargins(0, 0, 0, 0)
        self.residual = QComboBox()
        for key, label in templates.RESIDUALS.items():
            self.residual.addItem(label, key)
        self.residual.setToolTip("A lower panel comparing the other series with series 1")
        self.residual.currentIndexChanged.connect(self._controls_changed)
        f.addRow("Lower panel", self.residual)

        def pair(tip: str):
            a, b = QLineEdit(), QLineEdit()
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            for e, ph in ((a, "from"), (b, "to")):
                e.setPlaceholderText(ph)
                e.setToolTip(tip)
                e.editingFinished.connect(self._controls_changed)
                row.addWidget(e)
            return a, b, row

        self.y2label = QLineEdit()
        self.y2label.setPlaceholderText("automatic")
        self.y2label.setToolTip("Label of the right-hand axis")
        self.y2label.editingFinished.connect(self._controls_changed)
        self.y2scale = QComboBox()
        for text, key in (("linear", "linear"), ("log", "log")):
            self.y2scale.addItem(text, key)
        self.y2scale.currentIndexChanged.connect(self._controls_changed)
        r2 = QHBoxLayout()
        r2.setContentsMargins(0, 0, 0, 0)
        r2.addWidget(self.y2label, 1)
        r2.addWidget(self.y2scale)
        f.addRow("Right axis", r2)
        self.inset_x0, self.inset_x1, rx = pair("Range the inset shows; empty: all of its data")
        self.inset_y0, self.inset_y1, ry = pair("Range the inset shows; empty: all of its data")
        f.addRow("Inset x", rx)
        f.addRow("Inset y", ry)
        self.break_axis = QComboBox()
        for key, label in templates.BREAKS.items():
            self.break_axis.addItem(label, key)
        self.break_axis.currentIndexChanged.connect(self._controls_changed)
        self.break_a, self.break_b, rb = pair("The range the axis leaves out")
        rb.insertWidget(0, self.break_axis)
        f.addRow("Break", rb)
        v.addLayout(f)
        return w

    @staticmethod
    def _set_pair(a: QLineEdit, b: QLineEdit, v) -> None:
        v = v or [None, None]
        for e, x in zip((a, b), v):
            e.setText("" if x is None else f"{x:.6g}")

    def _page_grotrian(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        b = QPushButton("Levels and lines...")
        b.setToolTip("Choose the levels to draw, then the lines between them")
        b.clicked.connect(self._grotrian_dialog)
        self.grot_summary = QLabel("")
        self.grot_summary.setWordWrap(True)
        f.addRow(b)
        f.addRow(self.grot_summary)
        self.grot_unit = QComboBox()
        self.grot_unit.addItem("eV", "eV")
        self.grot_unit.addItem("cm⁻¹", "cm-1")
        self.grot_arrows = QComboBox()
        for text, key in (("Emission (down)", "emission"), ("Absorption (up)", "absorption"), ("No heads", "none")):
            self.grot_arrows.addItem(text, key)
        self.grot_colour = QComboBox()
        for text, key in (("By lower level", "lower"), ("By wavelength", "wavelength"), ("One colour", "single")):
            self.grot_colour.addItem(text, key)
        for c in (self.grot_unit, self.grot_arrows, self.grot_colour):
            c.currentIndexChanged.connect(self._controls_changed)
        f.addRow("Energy", self.grot_unit)
        f.addRow("Arrows", self.grot_arrows)
        f.addRow("Colour", self.grot_colour)
        self.grot_flags = {}
        for key, text in (("j_split", "Separate J columns"), ("wavelengths", "Wavelength labels"),
                          ("label_levels", "Level labels"), ("limit", "Ionization limit"),
                          ("width_by_A", "Line width follows A")):
            c = QCheckBox(text)
            c.toggled.connect(self._controls_changed)
            self.grot_flags[key] = c
            f.addRow(c)
        return w

    def _page_quantum_defects(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        row, self.qd_l = self._checks([(c, c) for c in "spdfg"], "Series to draw, one panel each")
        f.addRow("Series", row)
        self.qd_measured = QCheckBox("Measured (NIST) levels only")
        self.qd_fit = QCheckBox("Show the Ritz fit")
        for c in (self.qd_measured, self.qd_fit):
            c.toggled.connect(self._controls_changed)
            f.addRow(c)
        return w

    def _page_scaling(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.sc_qty = QComboBox()
        for key, (label, _stage, _y, _p) in templates.SCALING.items():
            self.sc_qty.addItem(label, key)
        self.sc_qty.currentIndexChanged.connect(self._controls_changed)
        f.addRow("Quantity", self.sc_qty)
        row, self.sc_l = self._checks([(c, c) for c in "spdf"])
        f.addRow("Series", row)
        self.sc_fit = QCheckBox("Fit a power law")
        self.sc_fit.toggled.connect(self._controls_changed)
        f.addRow(self.sc_fit)
        self.sc_from = QDoubleSpinBox()
        self.sc_from.setRange(1.0, 100.0)
        self.sc_from.setToolTip("Fit only n* at or above this")
        self.sc_from.valueChanged.connect(self._controls_changed)
        f.addRow("Fit from n*", self.sc_from)
        return w

    def _page_theory_vs_expt(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        row, self.tve_q = self._checks([(k, k.replace("tau_", "τ ").replace("alpha0", "α0"))
                                        for k in templates.QUANTITIES])
        f.addRow("Compare", row)
        note = QLabel("All alkali species with results in the output folder.")
        note.setWordWrap(True)
        f.addRow(note)
        return w

    def _page_alpha(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.al_states = QComboBox()
        for text, key in (("Ground and excited", "both"), ("Ground", "ground"), ("Excited", "excited")):
            self.al_states.addItem(text, key)
        self.al_states.currentIndexChanged.connect(self._controls_changed)
        f.addRow("States", self.al_states)
        self.al_min, self.al_max, self.al_ymax = QLineEdit(), QLineEdit(), QLineEdit()
        for e, ph in ((self.al_min, "400"), (self.al_max, "1600"), (self.al_ymax, "auto")):
            e.setPlaceholderText(ph)
            e.editingFinished.connect(self._controls_changed)
        f.addRow("λ range (nm)", self._pair(self.al_min, self.al_max))
        f.addRow("|α| up to", self.al_ymax)
        self.al_points = QSpinBox()
        self.al_points.setRange(200, 200_000)
        self.al_points.setSingleStep(1000)
        self.al_points.setValue(templates.ALPHA_POINTS)
        self.al_points.setToolTip("Points across the range; each resonance is also sampled finely")
        self.al_points.editingFinished.connect(self._controls_changed)
        f.addRow("Samples", self.al_points)
        self.al_magic = QCheckBox("Mark magic wavelengths")
        self.al_magic.toggled.connect(self._controls_changed)
        f.addRow(self.al_magic)
        self.al_form = f
        self.al_n0 = self._edit("1", "Lowest n of the excited-state lines whose magic wavelengths are shown")
        self.al_n1 = self._edit("n₀ + 5", "Highest n; raise it to see the crossings by the Rydberg lines")
        f.addRow("Lines n from, to", self._pair(self.al_n0, self.al_n1))
        self.al_show = self._combo(list(templates.MAGIC_SHOW.items()),
                                   "Which crossings to draw: cut near lines, all, or the practical ones")
        self.al_show.currentIndexChanged.connect(self._al_rows)
        f.addRow("Crossings", self.al_show)
        self.al_win0 = self._edit("none", "Drawn in full here; sweep it across the spectrum")
        self.al_win1 = self._edit("none", "Drawn in full here; sweep it across the spectrum")
        self.al_win_row = self._pair(self.al_win0, self.al_win1)
        f.addRow("Full window (nm)", self.al_win_row)
        self.al_cut = self._edit(f"{templates.MAGIC_CUTOFF_NM:g}", "Outside the window, cut ± this around each line")
        f.addRow("Cut (± nm)", self.al_cut)
        self.al_rmax = self._edit("from the stage", "Photon recoil heats atoms out of the trap no sooner than this "
                                  "(the same at any depth)")
        f.addRow("Heat-out time ≥ (s)", self.al_rmax)
        self.al_pmax = self._edit("no limit", "Trap photons the excited state absorbs per natural decay, 1 mK deep")
        f.addRow("Absorbed per decay ≤", self.al_pmax)
        self.al_wmin = self._edit("no limit", "Laser-frequency range over which |Δα| stays within 1 % of α, at least")
        f.addRow("Window ≥ (GHz)", self.al_wmin)
        self.al_marker = QComboBox()
        for text, m in MARKER_CHOICES[1:]:
            self.al_marker.addItem(text, m)
        self.al_fill = QComboBox()
        self.al_fill.addItem("open", "open")
        self.al_fill.addItem("filled", "filled")
        self.al_msize = QDoubleSpinBox()
        self.al_msize.setRange(0.0, 30.0)
        self.al_msize.setSingleStep(0.5)
        self.al_msize.setDecimals(1)
        self.al_msize.setSpecialValueText("auto")
        self.al_msize.setSuffix(" pt")
        self.al_mcolour = QComboBox()
        for text, c in (("black", "k"), *((f"palette {i + 1}", f"C{i}") for i in range(8)), ("grey", "0.5")):
            self.al_mcolour.addItem(text, c)
        for c in (self.al_marker, self.al_fill, self.al_mcolour):
            c.currentIndexChanged.connect(self._controls_changed)
        self.al_msize.valueChanged.connect(self._controls_changed)
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        for c, k in ((self.al_marker, 2), (self.al_fill, 1), (self.al_msize, 1)):
            lay.addWidget(c, k)
        f.addRow("Marker", row)
        f.addRow("Marker colour", self.al_mcolour)
        self._al_rows()
        return w

    def _al_rows(self, *_a) -> None:
        """Only the controls of the chosen crossings display."""
        show = self.al_show.currentData()
        for widget, on in ((self.al_win_row, show == "range"), (self.al_cut, show == "range"),
                           (self.al_rmax, show == "practical"), (self.al_pmax, show == "practical"),
                           (self.al_wmin, show == "practical")):
            self.al_form.setRowVisible(widget, on)

    def _combo(self, items, tip: str = "") -> QComboBox:
        c = QComboBox()
        for key, label in items:
            c.addItem(label, key)
        c.setToolTip(tip)
        c.currentIndexChanged.connect(self._controls_changed)
        return c

    def _check(self, text: str, tip: str = "") -> QCheckBox:
        c = QCheckBox(text)
        c.setToolTip(tip)
        c.toggled.connect(self._controls_changed)
        return c

    def _edit(self, placeholder: str, tip: str = "") -> QLineEdit:
        e = QLineEdit()
        e.setPlaceholderText(placeholder)
        e.setToolTip(tip)
        e.editingFinished.connect(self._controls_changed)
        return e

    def _checklist(self, tip: str = "") -> QListWidget:
        w = QListWidget()
        w.setMaximumHeight(110)
        w.setToolTip(tip)
        w.itemChanged.connect(self._controls_changed)
        return w

    def _page_contributions(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.co_qty = self._combo([("alpha0", "α0 (static)"), ("C6", "C6")])
        self.co_view = self._combo(list(templates.CONTRIB_VIEWS.items()))
        self.co_top = QSpinBox()
        self.co_top.setRange(1, 30)
        self.co_top.setToolTip("Lines shown one by one; the rest are summed")
        self.co_top.valueChanged.connect(self._controls_changed)
        self.co_merge = self._check("Add the J components (np1/2 + np3/2)")
        self.co_log = self._check("Log scale", "Contributions by line")
        self.co_rel = self._check("Relative to the total", "Convergence")
        f.addRow("Quantity", self.co_qty)
        f.addRow("Show", self.co_view)
        f.addRow("Lines shown", self.co_top)
        for c in (self.co_merge, self.co_log, self.co_rel):
            f.addRow(c)
        return w

    def _page_branching(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.br_state = self._combo([], "Decaying level")
        self.br_top = QSpinBox()
        self.br_top.setRange(1, 30)
        self.br_top.setToolTip("Channels shown one by one; the rest are summed")
        self.br_top.valueChanged.connect(self._controls_changed)
        self.br_log = self._check("Log scale")
        f.addRow("Level", self.br_state)
        f.addRow("Channels shown", self.br_top)
        f.addRow(self.br_log)
        return w

    def _page_cascade(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.ca_state = self._combo([], "Level the cascade starts from")
        self.ca_min = QDoubleSpinBox()
        self.ca_min.setRange(0.01, 50.0)
        self.ca_min.setDecimals(2)
        self.ca_min.setSuffix(" %")
        self.ca_min.setToolTip("Leave out levels and lines carrying fewer atoms")
        self.ca_min.valueChanged.connect(self._controls_changed)
        self.ca_unit = self._combo([("eV", "eV"), ("cm-1", "cm⁻¹")])
        self.ca_labels = self._check("Wavelengths on the lines")
        f.addRow("Start level", self.ca_state)
        f.addRow("Smallest share", self.ca_min)
        f.addRow("Energy", self.ca_unit)
        f.addRow(self.ca_labels)
        return w

    def _page_oscillator_density(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.od_unit = self._combo([("eV", "eV"), ("au", "a.u. (Hartree)")])
        self.od_merge = self._check("Add the J components")
        self.od_min, self.od_max = self._edit("auto"), self._edit("auto")
        f.addRow("Energy", self.od_unit)
        f.addRow("ε range", self._pair(self.od_min, self.od_max))
        f.addRow(self.od_merge)
        return w

    def _page_blackbody(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.bb_view = self._combo(list(templates.BBR_VIEWS.items()))
        row, self.bb_l = self._checks([(c, c) for c in "spdf"], "Series (shifts, rates; the map: the first ticked)")
        self.bb_qty = self._combo(list(templates.BBR_MAP_QTY.items()))
        self.bb_sweep_names: list[str] = []          # levels the stage swept in temperature
        self.bb_sweep_on: list[str] = []
        self.bb_states_btn = QPushButton("Levels...")
        self.bb_states_btn.setToolTip("Levels drawn against temperature")
        self.bb_states_btn.clicked.connect(self._bb_levels_dialog)
        self.bb_states_lbl = QLabel("")
        f.addRow("Show", self.bb_view)
        self.bb_series_row = row
        f.addRow("Series", row)
        f.addRow("Plot", self.bb_qty)
        f.addRow(self._pair(self.bb_states_btn, self.bb_states_lbl))
        self.bb_form = f
        self.bb_view.currentIndexChanged.connect(self._bb_rows)
        return w

    def _bb_rows(self, *_a) -> None:
        """Only the controls of the chosen view."""
        view = self.bb_view.currentData()
        sweep = view == "temperature"
        for widget, show in ((self.bb_series_row, not sweep), (self.bb_qty, view in ("temperature", "map")),
                             (self.bb_states_btn.parentWidget(), sweep)):
            self.bb_form.setRowVisible(widget, show)
        n, on = len(self.bb_sweep_names), len(self.bb_sweep_on)
        self.bb_states_lbl.setText(f"{on} of {n} levels" if n else "no temperature sweep")
        self.bb_states_btn.setEnabled(bool(n))

    def _bb_levels_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Temperature sweep: levels")
        v = QVBoxLayout(dlg)
        info = QLabel("The Blackbody stage follows these levels from 50 to 500 K. Ticked levels are drawn.")
        info.setWordWrap(True)
        v.addWidget(info)
        lst = QListWidget()
        for st in self.bb_sweep_names:
            it = QListWidgetItem(st)
            it.setData(Qt.UserRole, st)
            it.setCheckState(Qt.Checked if st in self.bb_sweep_on else Qt.Unchecked)
            lst.addItem(it)
        v.addWidget(lst)
        row = QHBoxLayout()
        for text, state in (("All", Qt.Checked), ("None", Qt.Unchecked)):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, st_=state: [lst.item(i).setCheckState(st_) for i in range(lst.count())])
            row.addWidget(b)
        row.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(dlg.accept)
        box.rejected.connect(dlg.reject)
        row.addWidget(box)
        v.addLayout(row)
        if dlg.exec() == QDialog.Accepted:
            self.bb_sweep_on = self._checked(lst)
            self._bb_rows()
            self._controls_changed()

    def _page_breit_rabi(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.bra_state = self._combo([], "Level with hyperfine constants")
        self.bra_bmax = self._edit("auto", "Field range (G)")
        self.bra_unit = self._combo([("", "automatic"), ("MHz", "MHz"), ("GHz", "GHz")])
        f.addRow("Level", self.bra_state)
        f.addRow("B up to (G)", self.bra_bmax)
        f.addRow("Energy", self.bra_unit)
        return w

    def _page_feshbach(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.fe_channels = self._checklist("Entrance channels to draw")
        self.fe_bmin, self.fe_bmax = self._edit("auto"), self._edit("auto")
        self.fe_ymax = self._edit("20000", "|a| range (a0)")
        self.fe_poles = self._check("Mark the resonances")
        f.addRow("Channels", self.fe_channels)
        f.addRow("B range (G)", self._pair(self.fe_bmin, self.fe_bmax))
        f.addRow("|a| up to (a0)", self.fe_ymax)
        f.addRow(self.fe_poles)
        return w

    def _page_radial(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.ra_states = self._edit("ground, lowest p and d", "Levels, e.g. 5s, 5p3/2, 30s")
        self.ra_qty = self._combo([("u", "u(r) = r R(r)"), ("density", "r² |R(r)|²")])
        self.ra_scale = self._combo([("log", "log r"), ("linear", "linear r")])
        self.ra_core = self._check("Shade the ionic core")
        f.addRow("Levels", self.ra_states)
        f.addRow("Plot", self.ra_qty)
        f.addRow("Axis", self.ra_scale)
        f.addRow(self.ra_core)
        return w

    def _page_spectrum(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.spc_view = self._combo(list(templates.SPECTRUM_VIEWS.items()))
        self.spc_qty = self._combo([("", "automatic")] + list(templates.SPECTRUM_QTY.items()))
        self.spc_line = self._combo([("D2", "D2"), ("D1", "D1")])
        self.spc_T, self.spc_L = self._edit("cell default", "Gas temperature (K)"), self._edit("5", "Cell length (cm)")
        self.spc_texc = self._edit("5000", "Excitation temperature of the upper levels (K), or inf")
        self.spc_min, self.spc_max = self._edit("auto"), self._edit("auto")
        self.spc_inst = self._edit("0", "Instrument resolution, Gaussian FWHM (pm)")
        self.spc_air = self._check("Air wavelengths")
        self.spc_sticks = self._check("Hyperfine components as sticks")
        self.spc_labels = QSpinBox()
        self.spc_labels.setRange(0, 20)
        self.spc_labels.setToolTip("Strongest lines labelled")
        self.spc_labels.valueChanged.connect(self._controls_changed)
        self.spc_form = f
        for label, widget in (("Show", self.spc_view), ("Plot", self.spc_qty), ("Line", self.spc_line),
                              ("T (K)", self.spc_T), ("Cell (cm)", self.spc_L), ("T exc (K)", self.spc_texc),
                              ("λ range (nm)", self._pair(self.spc_min, self.spc_max)),
                              ("Instrument (pm)", self.spc_inst), ("Labels", self.spc_labels)):
            f.addRow(label, widget)
        f.addRow(self.spc_air)
        f.addRow(self.spc_sticks)
        self.spc_view.currentIndexChanged.connect(self._spc_rows)
        return w

    def _spc_rows(self, *_a) -> None:
        v = self.spc_view.currentData()
        rows = {self.spc_line: v == "dlines", self.spc_texc: v == "emission", self.spc_sticks: v == "dlines",
                self.spc_min.parentWidget(): v != "dlines", self.spc_labels: v != "dlines", self.spc_air: v != "dlines",
                self.spc_qty: v != "emission", self.spc_L: v != "emission"}
        for widget, show in rows.items():
            self.spc_form.setRowVisible(widget, show)

    def _page_trap(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.tr_I = self._edit("50", "Intensity (kW/cm²)")
        self.tr_min, self.tr_max = self._edit("400"), self._edit("1600")
        self.tr_umax = self._edit("auto", "|U| range (µK)")
        self.tr_res = self._check("Mark the strongest lines")
        f.addRow("I (kW/cm²)", self.tr_I)
        f.addRow("λ range (nm)", self._pair(self.tr_min, self.tr_max))
        f.addRow("|U| up to (µK)", self.tr_umax)
        f.addRow(self.tr_res)
        return w

    # ----- data-dependent lists (levels, channels): filled for the figure's species
    def _fill_choices(self) -> None:
        r = self.recipe
        t, sp, p = r["template"], r.get("species") or "", r.get("params") or {}
        if not (sp and self.data_dir):
            return
        was, self._building = self._building, True
        try:
            with catalog.using_inputs(sp, r.get("inputs") or {}):
                if t == "branching":
                    self.br_state.clear()
                    self.br_state.addItem("automatic", "")
                    for st in templates.lifetime_states(sp, self.data_dir):
                        self.br_state.addItem(st, st)
                    self.br_state.setCurrentIndex(max(0, self.br_state.findData(p.get("state", ""))))
                elif t == "cascade":
                    self.ca_state.clear()
                    self.ca_state.addItem("automatic", "")
                    for st in templates.cascade_states(sp, self.data_dir):
                        self.ca_state.addItem(st, st)
                    self.ca_state.setCurrentIndex(max(0, self.ca_state.findData(p.get("state", ""))))
                elif t == "breit_rabi":
                    self.bra_state.clear()
                    for st in templates.hyperfine_states(sp, self.data_dir):
                        self.bra_state.addItem(st, st)
                    self.bra_state.setCurrentIndex(max(0, self.bra_state.findData(p.get("state", ""))))
                elif t == "feshbach":
                    self.fe_channels.clear()
                    chans = templates.feshbach_channels(sp, self.data_dir)
                    on = p.get("channels") or chans[:1]
                    for ch in chans:
                        it = QListWidgetItem(ch)
                        it.setData(Qt.UserRole, ch)
                        it.setCheckState(Qt.Checked if ch in on else Qt.Unchecked)
                        self.fe_channels.addItem(it)
                elif t == "blackbody":
                    d = catalog.load_json(sp, "blackbody", self.data_dir) or {}
                    self.bb_sweep_names = sorted(((d.get("rates") or {}).get("T_sweep") or {}).get("states") or {},
                                                 key=templates.sweep_order)
                    on = p.get("states") or self.bb_sweep_names
                    self.bb_sweep_on = [st for st in self.bb_sweep_names if st in on]
                    self._bb_rows()
        finally:
            self._building = was

    @staticmethod
    def _checked(w: QListWidget) -> list[str]:
        return [w.item(i).data(Qt.UserRole) for i in range(w.count()) if w.item(i).checkState() == Qt.Checked]

    # ------------------------------------------------------------------ recipe <-> controls
    def _load_controls(self) -> None:
        """Show self.recipe in the controls."""
        self._building = True
        try:
            r = self.recipe
            self.template.setCurrentIndex(max(0, self.template.findData(r["template"])))
            self._show_page(r["template"])
            self.params_box.setTitle("Data" if r["template"] != "builder" else "Series")
            p = r.get("params") or {}
            a, z, st = r["axes"], r["size"], r["style"]
            self.xlabel.setText(a.get("xlabel") or "")
            self.ylabel.setText(a.get("ylabel") or "")
            self.title_edit.setText(a.get("title") or "")
            self.xscale.setCurrentIndex(max(0, self.xscale.findData(a.get("xscale") or "")))
            self.yscale.setCurrentIndex(max(0, self.yscale.findData(a.get("yscale") or "")))
            for w, k in ((self.xmin, "xmin"), (self.xmax, "xmax"), (self.ymin, "ymin"), (self.ymax, "ymax")):
                w.setText("" if a.get(k) is None else f"{a[k]:g}")
            self.legend.setCurrentIndex(max(0, self.legend.findData(a.get("legend", "auto"))))
            self.legend_loc.setCurrentText(a.get("legend_loc", "best"))
            self.legend_ncol.setValue(int(a.get("legend_ncol") or 1))
            self.legend_frame.setChecked(bool(a.get("legend_frame")))
            self.journal.setCurrentIndex(max(0, self.journal.findData(z.get("journal", "aps"))))
            self.columns.setCurrentIndex(max(0, self.columns.findData(int(z.get("columns") or 1))))
            self.aspect.setValue(float(z.get("aspect") or 0.0))
            self.width_cm.setValue(float(z.get("width_cm") or 0.0))
            self.height_cm.setValue(float(z.get("height_cm") or 0.0))
            self.palette.setCurrentIndex(max(0, self.palette.findData(st.get("palette"))))
            self.font_box.setCurrentIndex(max(0, self.font_box.findData(st.get("font"))))
            self.font_size.setValue(float(st.get("font_size", 8)))
            self.usetex.setChecked(bool(st.get("usetex")))
            self.ticks_in.setChecked(bool(st.get("ticks_in", True)))
            self.mirror.setChecked(bool(st.get("mirror_ticks", True)))
            self.minor.setChecked(bool(st.get("minor_ticks", True)))
            self.grid.setChecked(bool(st.get("grid")))
            self.panel_labels.setChecked(bool(st.get("panel_labels", True)))
            self.line_width.setValue(float(st.get("line_width", 1.0)))
            self.marker_size.setValue(float(st.get("marker_size", 3.5)))
            # template pages
            self.series_editor.set_series(p.get("series") or [])
            self.residual.setCurrentIndex(max(0, self.residual.findData(p.get("residual", ""))))
            self.y2label.setText(p.get("y2label", "") or "")
            self.y2scale.setCurrentIndex(max(0, self.y2scale.findData(p.get("y2scale") or "linear")))
            self._set_pair(self.inset_x0, self.inset_x1, p.get("inset_xlim"))
            self._set_pair(self.inset_y0, self.inset_y1, p.get("inset_ylim"))
            self.break_axis.setCurrentIndex(max(0, self.break_axis.findData(p.get("break_axis", "") or "")))
            self._set_pair(self.break_a, self.break_b, p.get("break_at"))
            self.grot_unit.setCurrentIndex(max(0, self.grot_unit.findData(p.get("energy_unit", "eV"))))
            self.grot_arrows.setCurrentIndex(max(0, self.grot_arrows.findData(p.get("arrows", "emission"))))
            self.grot_colour.setCurrentIndex(max(0, self.grot_colour.findData(p.get("color_by", "lower"))))
            for k, c in self.grot_flags.items():
                c.setChecked(bool(p.get(k, True)))
            self._grot_summary()
            want = p.get("l") or ["s", "p", "d", "f"]
            for k, c in self.qd_l.items():
                c.setChecked(k in want)
            self.qd_measured.setChecked(bool(p.get("measured_only", True)))
            self.qd_fit.setChecked(bool(p.get("fit", True)))
            self.sc_qty.setCurrentIndex(max(0, self.sc_qty.findData(p.get("quantity", "lifetime"))))
            want = p.get("l") or ["s", "p", "d"]
            for k, c in self.sc_l.items():
                c.setChecked(k in want)
            self.sc_fit.setChecked(bool(p.get("fit", True)))
            self.sc_from.setValue(float(p.get("fit_from", 10)))
            want = p.get("quantities") or list(templates.QUANTITIES)
            for k, c in self.tve_q.items():
                c.setChecked(k in want)
            self.al_states.setCurrentIndex(max(0, self.al_states.findData(p.get("states", "both"))))
            self.al_min.setText(f"{p['wl_min']:g}" if p.get("wl_min") else "")
            self.al_max.setText(f"{p['wl_max']:g}" if p.get("wl_max") else "")
            self.al_ymax.setText(f"{p['ymax']:g}" if p.get("ymax") else "")
            self.al_magic.setChecked(bool(p.get("magic", True)))
            self.al_points.setValue(int(p.get("points") or templates.ALPHA_POINTS))
            self.al_marker.setCurrentIndex(max(0, self.al_marker.findData(p.get("magic_marker") or "o")))
            self.al_fill.setCurrentIndex(max(0, self.al_fill.findData(p.get("magic_fill", "open"))))
            self.al_msize.setValue(float(p.get("magic_size") or 0.0))
            self.al_mcolour.setCurrentIndex(max(0, self.al_mcolour.findData(p.get("magic_colour") or "k")))
            self.al_show.setCurrentIndex(max(0, self.al_show.findData(p.get("magic_show") or "range")))
            for e, k in ((self.al_win0, "range_min"), (self.al_win1, "range_max"), (self.al_cut, "cutoff_nm"),
                         (self.al_rmax, "heating_min"), (self.al_pmax, "excite_max"),
                         (self.al_wmin, "window_min"), (self.al_n0, "magic_n_min"), (self.al_n1, "magic_n_max")):
                e.setText(f"{p[k]:g}" if p.get(k) is not None else "")
            self._al_rows()
            self.co_qty.setCurrentIndex(max(0, self.co_qty.findData(p.get("quantity", "alpha0"))))
            self.co_view.setCurrentIndex(max(0, self.co_view.findData(p.get("view", "channels"))))
            self.co_top.setValue(int(p.get("top", 6)))
            self.co_merge.setChecked(bool(p.get("merge_j", True)))
            self.co_log.setChecked(bool(p.get("log", True)))
            self.co_rel.setChecked(bool(p.get("relative", False)))
            self.br_top.setValue(int(p.get("top", 8)))
            self.br_log.setChecked(bool(p.get("log", False)))
            self.ca_min.setValue(float(p.get("min_pct", 1.0)))
            self.ca_unit.setCurrentIndex(max(0, self.ca_unit.findData(p.get("energy_unit", "eV"))))
            self.ca_labels.setChecked(bool(p.get("line_labels", True)))
            self.od_unit.setCurrentIndex(max(0, self.od_unit.findData(p.get("energy_unit", "eV"))))
            self.od_merge.setChecked(bool(p.get("merge_j", False)))
            self.od_min.setText(f"{p['eps_min']:g}" if p.get("eps_min") is not None else "")
            self.od_max.setText(f"{p['eps_max']:g}" if p.get("eps_max") is not None else "")
            self.bb_view.setCurrentIndex(max(0, self.bb_view.findData(p.get("view", "shifts"))))
            want = p.get("l") or ["s", "p", "d"]
            for k, c in self.bb_l.items():
                c.setChecked(k in want)
            self.bb_qty.setCurrentIndex(max(0, self.bb_qty.findData(p.get("quantity", "tau"))))
            self._bb_rows()
            self.bra_bmax.setText(f"{p['B_max']:g}" if p.get("B_max") else "")
            self.bra_unit.setCurrentIndex(max(0, self.bra_unit.findData(p.get("unit", ""))))
            self.fe_bmin.setText(f"{p['B_min']:g}" if p.get("B_min") is not None else "")
            self.fe_bmax.setText(f"{p['B_max']:g}" if p.get("B_max") is not None else "")
            self.fe_ymax.setText(f"{p['ymax']:g}" if p.get("ymax") else "")
            self.fe_poles.setChecked(bool(p.get("poles", True)))
            self.ra_states.setText(", ".join(p.get("states") or []))
            self.ra_qty.setCurrentIndex(max(0, self.ra_qty.findData(p.get("quantity", "u"))))
            self.ra_scale.setCurrentIndex(max(0, self.ra_scale.findData(p.get("xscale", "log"))))
            self.ra_core.setChecked(bool(p.get("core", True)))
            self.spc_view.setCurrentIndex(max(0, self.spc_view.findData(p.get("view", "absorption"))))
            self.spc_qty.setCurrentIndex(max(0, self.spc_qty.findData(p.get("quantity") or "")))
            self.spc_line.setCurrentIndex(max(0, self.spc_line.findData(p.get("line", "D2"))))
            for e_, k in ((self.spc_T, "T_K"), (self.spc_L, "cell_cm"), (self.spc_texc, "T_exc"),
                          (self.spc_min, "wl_min"), (self.spc_max, "wl_max"), (self.spc_inst, "instrument_pm"),
                          (self.tr_I, "I_kWcm2"), (self.tr_min, "wl_min"), (self.tr_max, "wl_max"),
                          (self.tr_umax, "U_max")):
                v_ = p.get(k)
                e_.setText("" if v_ in (None, "") else (str(v_) if isinstance(v_, str) else f"{v_:g}"))
            self.spc_air.setChecked(bool(p.get("air", False)))
            self.spc_sticks.setChecked(bool(p.get("sticks", True)))
            self.spc_labels.setValue(int(p.get("labels", 6)))
            self.tr_res.setChecked(bool(p.get("resonances", True)))
            self._spc_rows()
        finally:
            self._building = False
        self._fill_choices()
        self._fill_inputs()

    # ------------------------------------------------------------------ inputs
    def _inputs_group(self) -> QGroupBox:
        self.inputs_box = QGroupBox("Input")
        self.inputs_form = QFormLayout(self.inputs_box)
        self._input_stages: list[str] = []
        self.input_boxes: dict[str, QComboBox] = {}
        return self.inputs_box

    def _fill_inputs(self) -> None:
        """One list per stage the figure reads: the usual output, the runs saved under a name, or a file."""
        r = self.recipe
        stages = templates.inputs_of(r["template"], r.get("params") or {})
        self._input_stages = stages
        while self.inputs_form.rowCount():
            self.inputs_form.removeRow(0)
        self.input_boxes = {}
        self.inputs_box.setVisible(bool(stages))
        sp = r.get("species") or ""
        chosen = r.setdefault("inputs", {})
        for st in stages:
            box = QComboBox()
            box.setToolTip(f"Which {st} result the figure reads")
            box.addItem("Usual output", "")
            for name in catalog.runs(sp, st, self.data_dir) if sp and self.data_dir else []:
                box.addItem(f"Saved as '{name}'", name)
            cur = chosen.get(st) or ""
            if cur and box.findData(cur) < 0:
                box.addItem(os.path.basename(cur) if os.path.isabs(cur) else f"'{cur}' (missing)", cur)
                if os.path.isabs(cur):
                    box.setItemData(box.count() - 1, cur, Qt.ToolTipRole)
            box.addItem("Other file...", OTHER_FILE)
            box.setCurrentIndex(max(0, box.findData(cur)))
            box.activated.connect(lambda _i, st=st: self._input_picked(st))
            self.inputs_form.addRow({"feshbach_out": "Feshbach", "hf_constants": "Hyperfine constants"}.get(
                st, st.replace("_", " ").capitalize()), box)
            self.input_boxes[st] = box

    def _input_picked(self, stage: str) -> None:
        box = self.input_boxes[stage]
        value = box.currentData()
        if value == OTHER_FILE:
            sp = self.recipe.get("species") or ""
            path, _ = QFileDialog.getOpenFileName(
                self, f"{sp} {stage} output", self.data_dir or self._start_dir(),
                f"{stage} output (*_{stage}*.json);;JSON (*.json)")
            if not path:
                self._fill_inputs()                   # back to what was chosen
                return
            value = os.path.abspath(path)
        self.recipe.setdefault("inputs", {})[stage] = value
        self._fill_inputs()
        self._fill_choices()
        self._schedule()

    # ------------------------------------------------------------------ zoom and pan
    def _fit_controls_width(self) -> None:
        """The controls panel is never narrower than its widest row plus the scroll bar, so no row runs
        under the scroll bar or off the edge."""
        sc = getattr(self, "controls_scroll", None)
        if sc is None or sc.widget() is None:
            return
        need = (sc.widget().minimumSizeHint().width() + sc.verticalScrollBar().sizeHint().width()
                + 2 * sc.frameWidth() + 2)
        if max(330, need) != sc.minimumWidth():
            sc.setMinimumWidth(max(330, need))

    def eventFilter(self, obj, event) -> bool:
        sc = getattr(self, "controls_scroll", None)
        if sc is not None and obj is sc.widget():
            if event.type() == QEvent.LayoutRequest:
                QTimer.singleShot(0, self._fit_controls_width)
            return super().eventFilter(obj, event)
        if obj is not self.preview or self.spec is None:
            return super().eventFilter(obj, event)
        t = event.type()
        if t == QEvent.Wheel:
            delta = event.angleDelta()
            steps = (delta.y() or delta.x()) / 120.0      # Alt + wheel arrives as a horizontal wheel on some systems
            mods = event.modifiers()
            if mods & Qt.ControlModifier:                  # the whole figure, as in the viewer and plot windows
                pos = self.preview_scroll.viewport().mapFromGlobal(event.globalPosition().toPoint())
                self._view_zoom_step(1 if steps > 0 else -1, pos)
                return True
            hit = self._hit(event.position().toPoint())
            if hit:
                i, ax, px, py = hit
                which = "x" if mods & Qt.ShiftModifier else "y" if mods & Qt.AltModifier else "xy"
                self._zoom(i, ax, px, py, 0.85 ** steps, which)
            return True
        if t == QEvent.MouseButtonPress and event.button() == Qt.MiddleButton:
            self._scroll_drag = event.globalPosition().toPoint()       # middle-drag moves a magnified figure
            return True
        if t == QEvent.MouseMove and self._scroll_drag is not None:
            p = event.globalPosition().toPoint()
            d = p - self._scroll_drag
            self._scroll_drag = p
            for bar, dv in ((self.preview_scroll.horizontalScrollBar(), d.x()),
                            (self.preview_scroll.verticalScrollBar(), d.y())):
                bar.setValue(bar.value() - dv)
            return True
        if t == QEvent.MouseButtonRelease and event.button() == Qt.MiddleButton:
            self._scroll_drag = None
            return True
        if t == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
            self._reset_view()
            return True
        if t == QEvent.MouseButtonPress:
            pos = event.position().toPoint()
            hit = self._hit(pos)
            if hit and event.button() == Qt.LeftButton:
                i, ax, px, py = hit
                self._pan = (i, ax, ax.transData.inverted().frozen(), ax.bbox.frozen(), px, py)
                self.preview.setCursor(Qt.ClosedHandCursor)
                return True
            if hit and event.button() == Qt.RightButton:
                self._box_start = pos
                self._band.setGeometry(QRect(pos, QSize()))
                self._band.show()
                return True
        if t == QEvent.MouseMove:
            pos = event.position().toPoint()
            if self._pan is not None:
                i, ax, inv, bb, px0, py0 = self._pan
                px, py = self._display(pos)
                ddx, ddy = px - px0, py - py0
                (x0, y0), (x1, y1) = inv.transform([(bb.x0 - ddx, bb.y0 - ddy), (bb.x1 - ddx, bb.y1 - ddy)])
                self._set_view(i, (x0, x1), (y0, y1), final=False)
                return True
            if self._box_start is not None:
                self._band.setGeometry(QRect(self._box_start, pos).normalized())
                return True
        if t == QEvent.MouseButtonRelease:
            if self._pan is not None:
                self._pan = None
                self.preview.unsetCursor()
                self._schedule()
                return True
            if self._box_start is not None:
                start, end = self._box_start, event.position().toPoint()
                self._box_start = None
                self._band.hide()
                hit = self._hit(start)
                if hit and abs(end.x() - start.x()) > 5 and abs(end.y() - start.y()) > 5:
                    i, ax, _px, _py = hit
                    inv = ax.transData.inverted()
                    a, b = self._display(start), self._display(end)
                    (x0, y0), (x1, y1) = inv.transform([(min(a[0], b[0]), min(a[1], b[1])),
                                                        (max(a[0], b[0]), max(a[1], b[1]))])
                    self._set_view(i, (x0, x1), (y0, y1))
                return True
        return super().eventFilter(obj, event)

    def _display(self, pos: QPoint) -> tuple[float, float]:
        """Figure (Agg) pixel coordinates, origin at the bottom left, of a point on the preview."""
        pix = self.preview.pixmap()
        r = pix.devicePixelRatio() or 1.0
        pw, ph = pix.width() / r, pix.height() / r
        ox, oy = (self.preview.width() - pw) / 2, (self.preview.height() - ph) / 2
        return (pos.x() - ox) * r, (ph - (pos.y() - oy)) * r

    def _hit(self, pos: QPoint):
        """(panel index, axes, x, y in figure pixels) of the panel under *pos*, or None."""
        pix = self.preview.pixmap()
        if self.spec is None or pix is None or pix.isNull():
            return None
        px, py = self._display(pos)
        panels = getattr(self.figure, "_qali_panels", None) or self.figure.axes[: len(self.spec.panels)]
        for i, ax in enumerate(panels):
            if ax is None:                                 # a broken axis: its two ranges are set by hand
                continue
            if any(c.bbox.contains(px, py) for c in ax.child_axes):
                return None                                # over an inset
            if ax.bbox.contains(px, py):
                return i, ax, px, py
        return None

    def _zoom(self, i: int, ax, px: float, py: float, factor: float, which: str) -> None:
        bb = ax.bbox
        (x0, y0), (x1, y1) = ax.transData.inverted().transform(
            [(px + (bb.x0 - px) * factor, py + (bb.y0 - py) * factor),
             (px + (bb.x1 - px) * factor, py + (bb.y1 - py) * factor)])
        self._set_view(i, (x0, x1) if "x" in which else None, (y0, y1) if "y" in which else None)

    def _set_view(self, i: int, xlim, ylim, final: bool = True) -> None:
        """New limits of panel *i*: drawn at once and kept in the recipe, so the export and the saved recipe
        show the same view."""
        if self.spec is None:
            return
        a = self.recipe["axes"]
        view = a.setdefault("view", {})
        axes = getattr(self.figure, "_qali_panels", None) or self.figure.axes
        if xlim is not None:
            xlim = [float(xlim[0]), float(xlim[1])]
            if i == 0 or self.spec.sharex:
                a["xmin"], a["xmax"] = xlim
            else:
                view.setdefault(str(i), {})["x"] = xlim
            for j in (range(len(self.spec.panels)) if self.spec.sharex else [i]):
                self.spec.panels[j].xlim = list(xlim)
            axes[i].set_xlim(*xlim)
        if ylim is not None:
            ylim = [float(ylim[0]), float(ylim[1])]
            if i == 0:
                a["ymin"], a["ymax"] = ylim
            else:
                view.setdefault(str(i), {})["y"] = ylim
            self.spec.panels[i].ylim = list(ylim)
            axes[i].set_ylim(*ylim)
        self._building = True
        for w, k in ((self.xmin, "xmin"), (self.xmax, "xmax"), (self.ymin, "ymin"), (self.ymax, "ymax")):
            w.setText("" if a.get(k) is None else f"{a[k]:.6g}")
        self._building = False
        S.clip_to_axes(axes[i])
        self._live.start()                         # draw now without recomputing the figure
        if final:
            self._schedule()                       # then redraw fully (log-axis ticks follow the range)

    def _reset_view(self) -> None:
        a = self.recipe["axes"]
        for k in ("xmin", "xmax", "ymin", "ymax"):
            a[k] = None
        a["view"] = {}
        self._building = True
        for w in (self.xmin, self.xmax, self.ymin, self.ymax):
            w.setText("")
        self._building = False
        self._schedule()

    def _show_page(self, key: str) -> None:
        """Show the template's page only (a hidden page takes no room)."""
        for k, page in self.param_pages.items():
            page.setVisible(k == key)

    def _params_from_controls(self) -> dict:
        t = self.recipe["template"]
        p = dict(self.recipe.get("params") or {})
        if t == "builder":
            p["series"] = self.series_editor.series()
            p["residual"] = self.residual.currentData()
            p["y2label"] = self.y2label.text().strip()
            p["y2scale"] = self.y2scale.currentData()
            p["inset_xlim"] = [_float_or_none(self.inset_x0.text()), _float_or_none(self.inset_x1.text())]
            p["inset_ylim"] = [_float_or_none(self.inset_y0.text()), _float_or_none(self.inset_y1.text())]
            p["break_axis"] = self.break_axis.currentData()
            p["break_at"] = [_float_or_none(self.break_a.text()), _float_or_none(self.break_b.text())]
        elif t == "grotrian":
            p["energy_unit"] = self.grot_unit.currentData()
            p["arrows"] = self.grot_arrows.currentData()
            p["color_by"] = self.grot_colour.currentData()
            for k, c in self.grot_flags.items():
                p[k] = c.isChecked()
        elif t == "quantum_defects":
            p["l"] = [k for k, c in self.qd_l.items() if c.isChecked()]
            p["measured_only"] = self.qd_measured.isChecked()
            p["fit"] = self.qd_fit.isChecked()
        elif t == "scaling":
            p["quantity"] = self.sc_qty.currentData()
            p["l"] = [k for k, c in self.sc_l.items() if c.isChecked()]
            p["fit"] = self.sc_fit.isChecked()
            p["fit_from"] = self.sc_from.value()
        elif t == "theory_vs_expt":
            p["quantities"] = [k for k, c in self.tve_q.items() if c.isChecked()]
        elif t == "contributions":
            p.update(quantity=self.co_qty.currentData(), view=self.co_view.currentData(), top=self.co_top.value(),
                     merge_j=self.co_merge.isChecked(), log=self.co_log.isChecked(), relative=self.co_rel.isChecked())
        elif t == "branching":
            p.update(state=self.br_state.currentData() or "", top=self.br_top.value(), log=self.br_log.isChecked())
        elif t == "cascade":
            p.update(state=self.ca_state.currentData() or "", min_pct=self.ca_min.value(),
                     energy_unit=self.ca_unit.currentData(), line_labels=self.ca_labels.isChecked())
        elif t == "oscillator_density":
            p.update(energy_unit=self.od_unit.currentData(), merge_j=self.od_merge.isChecked(),
                     eps_min=_float_or_none(self.od_min.text()), eps_max=_float_or_none(self.od_max.text()))
        elif t == "blackbody":
            p.update(view=self.bb_view.currentData(), l=[k for k, c in self.bb_l.items() if c.isChecked()],
                     quantity=self.bb_qty.currentData())
            if self.bb_sweep_names:
                p["states"] = list(self.bb_sweep_on)
        elif t == "breit_rabi":
            p.update(state=self.bra_state.currentData() or "", B_max=_float_or_none(self.bra_bmax.text()),
                     unit=self.bra_unit.currentData())
        elif t == "feshbach":
            if self.fe_channels.count():
                p["channels"] = self._checked(self.fe_channels)
            p.update(B_min=_float_or_none(self.fe_bmin.text()), B_max=_float_or_none(self.fe_bmax.text()),
                     ymax=_float_or_none(self.fe_ymax.text()), poles=self.fe_poles.isChecked())
        elif t == "radial":
            p["states"] = [x.strip() for x in self.ra_states.text().replace(";", ",").split(",") if x.strip()]
            p.update(quantity=self.ra_qty.currentData(), xscale=self.ra_scale.currentData(),
                     core=self.ra_core.isChecked())
        elif t == "spectrum":
            texc = self.spc_texc.text().strip()
            p.update(view=self.spc_view.currentData(), quantity=self.spc_qty.currentData() or None,
                     line=self.spc_line.currentData(), T_K=_float_or_none(self.spc_T.text()),
                     cell_cm=_float_or_none(self.spc_L.text()),
                     T_exc="inf" if texc.lower() in ("inf", "∞") else _float_or_none(texc),
                     wl_min=_float_or_none(self.spc_min.text()), wl_max=_float_or_none(self.spc_max.text()),
                     instrument_pm=_float_or_none(self.spc_inst.text()), air=self.spc_air.isChecked(),
                     sticks=self.spc_sticks.isChecked(), labels=self.spc_labels.value())
        elif t == "trap":
            p.update(I_kWcm2=_float_or_none(self.tr_I.text()), wl_min=_float_or_none(self.tr_min.text()),
                     wl_max=_float_or_none(self.tr_max.text()), U_max=_float_or_none(self.tr_umax.text()),
                     resonances=self.tr_res.isChecked())
        elif t == "alpha":
            p["states"] = self.al_states.currentData()
            p["wl_min"] = _float_or_none(self.al_min.text())
            p["wl_max"] = _float_or_none(self.al_max.text())
            p["ymax"] = _float_or_none(self.al_ymax.text())
            p["magic"] = self.al_magic.isChecked()
            p["points"] = self.al_points.value()
            p["magic_marker"] = self.al_marker.currentData()
            p["magic_fill"] = self.al_fill.currentData()
            p["magic_size"] = self.al_msize.value() or None
            p["magic_colour"] = self.al_mcolour.currentData()
            p["magic_show"] = self.al_show.currentData()
            p["range_min"] = _float_or_none(self.al_win0.text())
            p["range_max"] = _float_or_none(self.al_win1.text())
            p["cutoff_nm"] = _float_or_none(self.al_cut.text())
            p["heating_min"] = _float_or_none(self.al_rmax.text())
            p["window_min"] = _float_or_none(self.al_wmin.text())
            p["magic_n_min"] = _int_or_none(self.al_n0.text())
            p["magic_n_max"] = _int_or_none(self.al_n1.text())
            p["excite_max"] = _float_or_none(self.al_pmax.text())
        return p

    def _controls_changed(self, *_args) -> None:
        if self._building:
            return
        r = self.recipe
        r["params"] = self._params_from_controls()
        if templates.inputs_of(r["template"], r["params"]) != self._input_stages:
            self._fill_inputs()
        r["axes"].update({
            "xlabel": self.xlabel.text().strip(), "ylabel": self.ylabel.text().strip(),
            "title": self.title_edit.text().strip(),
            "xscale": self.xscale.currentData(), "yscale": self.yscale.currentData(),
            "xmin": _float_or_none(self.xmin.text()), "xmax": _float_or_none(self.xmax.text()),
            "ymin": _float_or_none(self.ymin.text()), "ymax": _float_or_none(self.ymax.text()),
            "legend": self.legend.currentData(), "legend_loc": self.legend_loc.currentText(),
            "legend_ncol": self.legend_ncol.value(), "legend_frame": self.legend_frame.isChecked(),
        })
        journal = self.journal.currentData()
        if journal != r["size"].get("journal"):
            j = JOURNALS[journal]
            self._building = True
            self.font_size.setValue(j.font_pt)                       # a new preset brings its font size
            self.font_box.setCurrentIndex(max(0, self.font_box.findData("sans" if j.family == "sans" else "cm")))
            self._building = False
        r["size"].update({"journal": journal, "columns": self.columns.currentData(),
                          "aspect": self.aspect.value() or None, "width_cm": self.width_cm.value() or None,
                          "height_cm": self.height_cm.value() or None})
        r["style"].update({
            "palette": self.palette.currentData(), "font": self.font_box.currentData(),
            "font_size": self.font_size.value(), "usetex": self.usetex.isChecked(),
            "ticks_in": self.ticks_in.isChecked(), "mirror_ticks": self.mirror.isChecked(),
            "minor_ticks": self.minor.isChecked(), "grid": self.grid.isChecked(),
            "panel_labels": self.panel_labels.isChecked(), "line_width": self.line_width.value(),
            "marker_size": self.marker_size.value(),
        })
        self._schedule()

    def _template_changed(self, _i: int) -> None:
        if self._building:
            return
        key = self.template.currentData()
        keep_size, keep_style = copy.deepcopy(self.recipe["size"]), copy.deepcopy(self.recipe["style"])
        species = self.species_box.currentData() or self.recipe.get("species") or self.species   # the one shown
        self.recipe = R.new_recipe(key, species)
        self.recipe["size"].update(keep_size)
        self.recipe["size"]["aspect"] = None            # each figure has its own shape
        self.recipe["style"].update(keep_style)
        self._load_controls()
        self._schedule()

    def _species_changed(self, _i: int) -> None:
        if self._building:
            return
        sp = self.species_box.currentData() or ""
        if sp and sp != self.recipe.get("species"):
            self.recipe["species"] = sp
            self.recipe["inputs"] = {}              # runs and files belong to one species
            for k in ("state", "states", "channels"):   # levels and channels belong to one species
                self.recipe["params"].pop(k, None)
            self._fill_inputs()
            self._fill_choices()
            if self.recipe["template"] == "grotrian":
                self.recipe["params"].pop("levels", None)
                self.recipe["params"].pop("lines", None)
                self._grot_summary()
            self._schedule()

    def _fill_species(self) -> None:
        self._building = True
        try:
            have = catalog.available_species(self.data_dir) if self.data_dir else []
            cur = self.recipe.get("species") or self.species
            if self.species and self.species in have and self.species != cur and not self.recipe.get("_pinned"):
                cur = self.species
            self.species_box.clear()
            for sp in have:
                self.species_box.addItem(sp, sp)
            if cur in have:
                self.species_box.setCurrentIndex(have.index(cur))
            elif have:
                cur = have[0]
            if (cur or "") != self.recipe.get("species"):
                self.recipe["inputs"] = {}          # runs and files belong to one species
            self.recipe["species"] = cur or ""
            self.series_editor.set_species(have, cur or "")
        finally:
            self._building = False

    # ------------------------------------------------------------------ drawing
    def _schedule(self) -> None:
        self._timer.start()

    def redraw(self) -> None:
        """Resolve the recipe and draw it; a missing stage output or a drawing error is shown as text."""
        self._stale = False
        if not self.data_dir:
            self._show_message("Choose an output folder with results.", error=True)
            return
        try:
            self.spec = R.resolve(self.recipe, self.data_dir)
        except templates.TemplateError as exc:
            self.spec = None
            self.preview.clear()
            self._show_message(str(exc), error=True)
            return
        try:
            S.draw(self.spec, self.figure, raster=True)
        except Exception as exc:
            self._show_message(S.explain(exc), error=True)
            return
        if self._render_preview():
            w, h = self.spec.width_in * 2.54, self.spec.height_in * 2.54
            self._show_message("\n".join([f"{w:.1f} × {h:.1f} cm."] + self.spec.notes), error=False)

    def _render_preview(self) -> bool:
        """Draw the figure as an image as large as the panel allows, at its printed proportions."""
        if self.spec is None:
            return False
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        vp = self.preview_scroll.viewport()
        avail_w = max(vp.width() - 12, 200)
        avail_h = max(vp.height() - 12, 150)
        ratio = self.devicePixelRatioF() or 1.0
        dpi = max(40.0, min(avail_w / self.spec.width_in, avail_h / self.spec.height_in, 400.0)) * self.view_zoom
        dpi = min(dpi, 1200.0)
        agg = FigureCanvasAgg(self.figure)
        self.figure.set_dpi(dpi * ratio)
        try:
            with S.style_context(self.spec.style, raster=True):
                agg.draw()
        except Exception as exc:
            self._show_message(S.explain(exc), error=True)
            return False
        w, h = agg.get_width_height()
        img = QImage(bytes(agg.buffer_rgba()), w, h, QImage.Format_RGBA8888).copy()
        pix = QPixmap.fromImage(img)
        pix.setDevicePixelRatio(ratio)
        self.preview.setPixmap(pix)
        fit = self.view_zoom == 1.0
        self.preview_scroll.setWidgetResizable(fit)
        if not fit:
            self.preview.resize(int(w / ratio), int(h / ratio))
        return True

    def _view_zoom_step(self, step: int, anchor: QPoint | None = None) -> None:
        """Magnify the whole figure (not its axis ranges): step +1 / −1, or 0 to fit the panel."""
        from .popout import next_zoom, show_zoom

        old = self.view_zoom
        self.view_zoom = 1.0 if step == 0 else max(1.0, next_zoom(self.view_zoom, step))
        if self.view_zoom == old or self.spec is None:
            return
        vp = self.preview_scroll.viewport()
        anchor = anchor or QPoint(vp.width() // 2, vp.height() // 2)
        h, v = self.preview_scroll.horizontalScrollBar(), self.preview_scroll.verticalScrollBar()
        fx = (h.value() + anchor.x()) / max(self.preview.width(), 1)
        fy = (v.value() + anchor.y()) / max(self.preview.height(), 1)
        self._render_preview()
        h.setValue(int(fx * self.preview.width() - anchor.x()))
        v.setValue(int(fy * self.preview.height() - anchor.y()))
        show_zoom(vp, self.view_zoom)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.spec is not None and self.isVisible():
            self._resize_timer.start()

    def _show_message(self, text: str, error: bool) -> None:
        self.message.setText(text)
        self.message.setStyleSheet("color: #d9534f;" if error else "")

    # ------------------------------------------------------------------ grotrian dialog
    def _grot_summary(self) -> None:
        p = self.recipe.get("params") or {}
        n_lv, n_ln = len(p.get("levels") or []), len(p.get("lines") or [])
        self.grot_summary.setText(
            (f"{n_lv} levels chosen" if n_lv else "Lowest levels (automatic)")
            + (f", {n_ln} lines." if n_ln else ", strongest lines between them."))

    def _grotrian_dialog(self) -> None:
        sp = self.recipe.get("species")
        try:
            lv, lines, IE = templates.grotrian_choices(sp, self.data_dir)
        except templates.TemplateError as exc:
            QMessageBox.information(self, "Grotrian diagram", str(exc))
            return
        p = self.recipe.get("params") or {}
        dlg = GrotrianDialog(lv, lines, IE, p.get("levels") or templates.default_grotrian_levels(lv, IE),
                             p.get("lines"), self)
        if dlg.exec() == QDialog.Accepted:
            self.recipe["params"]["levels"] = dlg.chosen_levels()
            self.recipe["params"]["lines"] = dlg.chosen_lines()
            self._grot_summary()
            self._schedule()

    # ------------------------------------------------------------------ files
    def _start_dir(self) -> str:
        return str(_settings().value(LAST_DIR_KEY, "") or os.path.dirname(self.data_dir) or os.getcwd())

    def _remember_dir(self, path: str) -> None:
        _settings().setValue(LAST_DIR_KEY, os.path.dirname(path))

    def _default_name(self, ext: str) -> str:
        sp = self.recipe.get("species") or "figure"
        return os.path.join(self._start_dir(), f"{sp}_{self.recipe['template']}.{ext}")

    def _export(self) -> None:
        if self.spec is None:
            QMessageBox.information(self, "Export", "There is no figure to export.")
            return
        export = load_defaults(_settings())["export"]
        fmt = export["format"]
        first = f"{S.SAVE_FORMATS[fmt]} (*.{fmt})"
        filters = ";;".join([first] + [f for f in EXPORT_FILTERS.split(";;") if f != first])
        path, _ = QFileDialog.getSaveFileName(self, "Export figure", self._default_name(fmt), filters)
        if not path:
            return
        if "." not in os.path.basename(path):
            path += f".{fmt}"
        try:
            S.save(self.spec, path, dpi=int(export.get("dpi") or 600))
        except (S.RenderError, ValueError, OSError) as exc:
            QMessageBox.warning(self, "Export", str(exc))
            return
        self._remember_dir(path)
        self._show_message(f"Saved {path}", error=False)

    def _script(self) -> None:
        if self.spec is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Python script", self._default_name("py"), "Python (*.py)")
        if not path:
            return
        if not path.endswith(".py"):
            path += ".py"
        out = os.path.splitext(os.path.basename(path))[0] + ".pdf"
        with open(path, "w", encoding="utf-8") as f:
            f.write(S.to_script(self.spec, out))
        self._remember_dir(path)
        self._show_message(f"Saved {path}; running it writes {out}.", error=False)

    def _csv(self) -> None:
        if self.spec is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Plotted data", self._default_name("csv"), "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            head, rows = S.data_table(self.spec)
            w.writerow(head)
            w.writerows(rows)
        self._remember_dir(path)
        self._show_message(f"Saved {path}", error=False)

    def _save_recipe(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save figure recipe", self._default_name("qfig"),
                                              "Qali figure (*.qfig)")
        if not path:
            return
        if not path.endswith(".qfig"):
            path += ".qfig"
        R.save_recipe(self.recipe, path)
        self._remember_dir(path)
        self._show_message(f"Saved {path}", error=False)

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open figure recipe", self._start_dir(),
                                              "Qali figure (*.qfig);;All files (*)")
        if not path:
            return
        try:
            self.open_recipe(path)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Open", str(exc))

    def open_recipe(self, path: str) -> None:
        self.recipe = R.load_recipe(path)
        self.recipe["_pinned"] = True                  # keep the recipe's species when the main selection moves
        self._fill_species()
        self._load_controls()
        self._remember_dir(path)
        self._schedule()

    def _save_defaults(self) -> None:
        settings = _settings()
        d = load_defaults(settings)
        d["style"].update(self.recipe["style"])
        d["size"].update({k: self.recipe["size"].get(k) for k in ("journal", "columns")})
        save_defaults(settings, d)
        self._show_message("New figures will start with this size and style.", error=False)


# ====================================================================== builder series editor
class SeriesEditor(QWidget):
    """The plot builder's series: each one a table, x and y (columns or expressions), a filter and a look."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._series: list[dict] = []
        self._species: list[str] = []
        self._default_species = ""
        self._loading = False
        self.tab: FiguresTab | None = parent if isinstance(parent, FiguresTab) else None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setMaximumHeight(90)
        v.addWidget(self.list)
        row = QHBoxLayout()
        for text, tip, slot in (("Add", "Add a series", self._add), ("Copy", "Copy the selected series", self._copy),
                                ("Remove", "Remove the selected series", self._remove)):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        v.addLayout(row)
        self.form_box = QWidget()
        f = QFormLayout(self.form_box)
        f.setContentsMargins(0, 0, 0, 0)
        self.species = QComboBox()
        self.table = QComboBox()
        self.table.setToolTip("A list in one of the stage outputs")
        self.x, self.y, self.yerr, self.z = QComboBox(), QComboBox(), QComboBox(), QComboBox()
        for c, tip in ((self.x, "Column or expression, e.g. n - 2.65 or log10(lifetime_ns)"),
                       (self.y, "Column or expression"), (self.yerr, "Column or expression; empty: no error bars"),
                       (self.z, "Column or expression mapped to colour")):
            c.setEditable(True)
            c.setInsertPolicy(QComboBox.NoInsert)
            c.setToolTip(tip)
        self.where = QLineEdit()
        self.where.setPlaceholderText("all rows")
        self.where.setToolTip("Rows to keep, e.g. l == 's' and n > 10")
        self.kind = QComboBox()
        for key, label in templates.KINDS.items():
            self.kind.addItem(label, key)
        self.on = QComboBox()
        for key, label in templates.SERIES_AXES.items():
            self.on.addItem(label, key)
        self.on.setToolTip("Left axis, a right-hand y axis, or an inset")
        self.label = QLineEdit()
        self.label.setPlaceholderText("no legend entry")
        self.colour = QComboBox()
        self.colour.addItem("automatic", "auto")
        for i in range(8):
            self.colour.addItem(f"palette {i + 1}", f"C{i}")
        self.colour.addItem("black", "k")
        self.colour.addItem("grey", "0.5")
        self.colour.addItem("other...", "pick")
        self.marker = QComboBox()
        for text, m in MARKER_CHOICES:
            self.marker.addItem(text, m)
        self.marker_fill = QComboBox()
        self.marker_fill.addItem("filled", "filled")
        self.marker_fill.addItem("open", "open")
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(0.0, 30.0)
        self.marker_size.setSingleStep(0.5)
        self.marker_size.setDecimals(1)
        self.marker_size.setSpecialValueText("auto")
        self.marker_size.setSuffix(" pt")
        self.marker_size.setToolTip("Marker size; auto: the style's size")
        self.marker_row = QWidget()
        _mr = QHBoxLayout(self.marker_row)
        _mr.setContentsMargins(0, 0, 0, 0)
        _mr.addWidget(self.marker, 2)
        _mr.addWidget(self.marker_fill, 1)
        _mr.addWidget(self.marker_size, 1)
        self.linestyle = QComboBox()
        for text, ls in (("solid", "-"), ("dashed", "--"), ("dash-dot", "-."), ("dotted", ":")):
            self.linestyle.addItem(text, ls)
        self.species.setToolTip("A species' stage outputs, or any JSON file")
        f.addRow("Source", self.species)
        f.addRow("Table", self.table)
        f.addRow("x", self.x)
        f.addRow("y", self.y)
        f.addRow("y error", self.yerr)
        f.addRow("z", self.z)
        f.addRow("Rows", self.where)
        f.addRow("Draw as", self.kind)
        f.addRow("Axes", self.on)
        self._form = f
        f.addRow("Legend", self.label)
        f.addRow("Colour", self.colour)
        f.addRow("Marker", self.marker_row)
        f.addRow("Line", self.linestyle)
        v.addWidget(self.form_box)
        self.help = QLabel("Expressions: + − * / **, log10, sqrt, exp, abs; filters with == != < >, and, or, & |. "
                           "A column whose name is not a word: col(\"25p3/2\").")
        self.help.setWordWrap(True)
        v.addWidget(self.help)
        self.list.currentRowChanged.connect(self._show)
        self.species.currentIndexChanged.connect(self._species_picked)
        self.table.currentIndexChanged.connect(self._table_picked)
        for c in (self.kind, self.marker, self.marker_fill, self.linestyle, self.on):
            c.currentIndexChanged.connect(self._edited)
        self.marker_size.valueChanged.connect(self._edited)
        self.kind.currentIndexChanged.connect(self._kind_rows)
        self.colour.currentIndexChanged.connect(self._colour_picked)
        for c in (self.x, self.y, self.yerr, self.z):
            c.currentTextChanged.connect(self._edited_later)
            c.lineEdit().editingFinished.connect(self._edited)
        for e in (self.where, self.label):
            e.editingFinished.connect(self._edited)
        self._custom_colour = ""
        self._update_enabled()
        self._kind_rows()

    def _kind_rows(self, *_a) -> None:
        """z for a heat map or contours; y error, legend, marker and line for the others."""
        is_map = self.kind.currentData() in templates.MAP_KINDS
        for w in (self.z,):
            self._form.setRowVisible(w, is_map)
        for w in (self.yerr, self.label, self.marker_row, self.linestyle, self.colour):
            self._form.setRowVisible(w, not is_map)

    def _edited_later(self, *_a) -> None:
        if not self._loading and not any(c.lineEdit().hasFocus() for c in (self.x, self.y, self.yerr, self.z)):
            self._edited()

    # ----- data in and out
    def set_species(self, species: list[str], default: str) -> None:
        self._species = species
        self._default_species = default
        self._fill_sources()

    @staticmethod
    def _source_of(s: dict) -> str:
        return f"file:{s['file']}" if s.get("file") else (s.get("species") or "")

    def _fill_sources(self, extra: str = "") -> None:
        """The species, the files the series read, and File... to add one."""
        self._loading, was = True, self._loading
        cur = extra or self.species.currentData()
        self.species.clear()
        for sp in self._species:
            self.species.addItem(sp, sp)
        files = [self._source_of(s) for s in self._series if s.get("file")]
        if extra.startswith("file:"):
            files.append(extra)
        for src in dict.fromkeys(files):
            self.species.addItem(os.path.basename(src[5:]), src)
            self.species.setItemData(self.species.count() - 1, src[5:], Qt.ToolTipRole)
        self.species.addItem("File...", OTHER_FILE)
        idx = self.species.findData(cur)
        self.species.setCurrentIndex(max(idx, 0))
        self._loading = was

    def _tables_of(self, source: str) -> list:
        if source.startswith("file:"):
            return catalog.file_tables(source[5:])
        return catalog.species_tables(source, self._data_dir()) if source else []

    def _table_of(self, source: str, key: str):
        if not (source and key):
            return None
        if source.startswith("file:"):
            return catalog.find_file_table(source[5:], key)
        return catalog.find_table(source, self._data_dir(), key)

    def set_series(self, series: list[dict]) -> None:
        self._series = [dict(s) for s in series]
        self._fill_sources()
        self._loading = True
        self.list.clear()
        for s in self._series:
            self.list.addItem(self._title(s))
        self._loading = False
        if self._series:
            self.list.setCurrentRow(0)
        self._update_enabled()

    def series(self) -> list[dict]:
        return [dict(s) for s in self._series]

    @staticmethod
    def _title(s: dict) -> str:
        return s.get("label") or f"{s.get('y', '?')} vs {s.get('x', '?')}"

    def _update_enabled(self) -> None:
        self.form_box.setEnabled(bool(self._series) and self.list.currentRow() >= 0)

    def _data_dir(self) -> str:
        return self.tab.data_dir if self.tab else ""

    # ----- list buttons
    def _add(self) -> None:
        sp = self._default_species or (self._species[0] if self._species else "")
        tables = catalog.species_tables(sp, self._data_dir()) if sp else []
        first = next((t for t in tables if t.key == "lifetimes:lifetimes"), tables[0] if tables else None)
        s = {"species": sp, "table": first.key if first else "", "x": "n", "y": "lifetime_ns", "kind": "markers"}
        if first and first.key != "lifetimes:lifetimes":
            nums = first.numeric_columns()
            s["x"], s["y"] = (nums + ["", ""])[:2]
        self._series.append(s)
        self.list.addItem(self._title(s))
        self.list.setCurrentRow(len(self._series) - 1)
        self._update_enabled()
        self.changed.emit()

    def _copy(self) -> None:
        i = self.list.currentRow()
        if 0 <= i < len(self._series):
            s = dict(self._series[i])
            s["color"] = "auto"
            self._series.append(s)
            self.list.addItem(self._title(s))
            self.list.setCurrentRow(len(self._series) - 1)
            self.changed.emit()

    def _remove(self) -> None:
        i = self.list.currentRow()
        if 0 <= i < len(self._series):
            del self._series[i]
            self.list.takeItem(i)
            self._update_enabled()
            self.changed.emit()

    # ----- form
    def _show(self, i: int) -> None:
        self._update_enabled()
        if not (0 <= i < len(self._series)):
            return
        s = self._series[i]
        self._loading = True
        try:
            src = self._source_of(s) or self._default_species
            if self.species.findData(src) < 0:
                self._fill_sources(src)
            self.species.setCurrentIndex(max(0, self.species.findData(src)))
            self._fill_tables(src, s.get("table", ""))
            self._fill_columns(src, s.get("table", ""))
            self.x.setCurrentText(s.get("x", ""))
            self.y.setCurrentText(s.get("y", ""))
            self.yerr.setCurrentText(s.get("yerr", "") or "")
            self.z.setCurrentText(s.get("z", "") or "")
            self.where.setText(s.get("where", "") or "")
            self.kind.setCurrentIndex(max(0, self.kind.findData(s.get("kind", "markers"))))
            self.on.setCurrentIndex(max(0, self.on.findData(s.get("on", "") or "")))
            self._kind_rows()
            self.label.setText(s.get("label", "") or "")
            col = s.get("color") or "auto"
            idx = self.colour.findData(col)
            if idx < 0:
                self._custom_colour = col
                idx = self.colour.findData("pick")
            self.colour.setCurrentIndex(idx)
            self.marker.setCurrentIndex(max(0, self.marker.findData(s.get("marker", ""))))
            self.marker_fill.setCurrentIndex(max(0, self.marker_fill.findData(s.get("marker_fill", "filled"))))
            self.marker_size.setValue(float(s.get("marker_size") or 0.0))
            self.linestyle.setCurrentIndex(max(0, self.linestyle.findData(s.get("linestyle", "-"))))
        finally:
            self._loading = False

    def _fill_tables(self, source: str, key: str) -> None:
        self.table.clear()
        for t in self._tables_of(source):
            self.table.addItem(t.title, t.key)
        self.table.setCurrentIndex(max(0, self.table.findData(key)))

    def _fill_columns(self, source: str, key: str) -> None:
        t = self._table_of(source, key)
        cols = list(t.columns) if t else []
        for c in (self.x, self.y, self.yerr, self.z):
            text = c.currentText()
            c.blockSignals(True)
            c.clear()
            if c is self.yerr:
                c.addItem("")
            c.addItems(cols)
            c.setCurrentText(text)
            c.blockSignals(False)

    def _species_picked(self, _i: int) -> None:
        if self._loading:
            return
        sp = self.species.currentData() or ""
        if sp == OTHER_FILE:
            path, _ = QFileDialog.getOpenFileName(self, "JSON file to plot from", self._data_dir(), "JSON (*.json)")
            i = self.list.currentRow()
            prev = self._source_of(self._series[i]) if 0 <= i < len(self._series) else ""
            self._fill_sources(f"file:{os.path.abspath(path)}" if path else prev)
            sp = self.species.currentData() or ""
            if not path:
                return
        self._loading = True
        self._fill_tables(sp, self._series[self.list.currentRow()].get("table", "") if self._series else "")
        self._fill_columns(sp, self.table.currentData() or "")
        self._loading = False
        self._edited()

    def _table_picked(self, _i: int) -> None:
        if self._loading:
            return
        self._loading = True
        self._fill_columns(self.species.currentData() or "", self.table.currentData() or "")
        self._loading = False
        self._edited()

    def _colour_picked(self, _i: int) -> None:
        if self._loading:
            return
        if self.colour.currentData() == "pick":
            c = QColorDialog.getColor(QColor(self._custom_colour or "#000000"), self, "Series colour")
            if c.isValid():
                self._custom_colour = c.name()
        self._edited()

    def _edited(self, *_a) -> None:
        if self._loading:
            return
        i = self.list.currentRow()
        if not (0 <= i < len(self._series)):
            return
        col = self.colour.currentData()
        src = self.species.currentData() or ""
        self._series[i] = {
            "species": "" if src.startswith("file:") else src, "table": self.table.currentData() or "",
            **({"file": src[5:]} if src.startswith("file:") else {}),
            "x": self.x.currentText().strip(), "y": self.y.currentText().strip(),
            "yerr": self.yerr.currentText().strip(), "where": self.where.text().strip(),
            "kind": self.kind.currentData(), "label": self.label.text().strip(),
            **({"z": self.z.currentText().strip()} if self.kind.currentData() in templates.MAP_KINDS else {}),
            **({"on": self.on.currentData()} if self.on.currentData() else {}),
            "color": self._custom_colour if col == "pick" else col,
            "marker": self.marker.currentData(), "linestyle": self.linestyle.currentData(),
            "marker_fill": self.marker_fill.currentData(), "marker_size": self.marker_size.value(),
        }
        self.list.item(i).setText(self._title(self._series[i]))
        self.changed.emit()


# ====================================================================== Grotrian level / line dialog
class GrotrianDialog(QDialog):
    """Choose the levels of a Grotrian diagram, then the lines between them."""

    def __init__(self, levels: list, lines: list[dict], IE: float, chosen: list[str], chosen_lines, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Grotrian diagram: levels and lines")
        self.resize(820, 560)
        self.levels, self.all_lines, self.IE = levels, lines, IE
        self._line_keys = set(chosen_lines or [])
        self._auto_lines = not chosen_lines
        v = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        v.addWidget(split, 1)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("1. Levels"))
        quick = QHBoxLayout()
        self.n_max = QSpinBox()
        self.n_max.setRange(1, 200)
        self.n_max.setValue(max((x.n for x in levels[:1]), default=1) + 3)
        self.l_max = QComboBox()
        for i, c in enumerate("spdfg"):
            self.l_max.addItem(c, i)
        self.l_max.setCurrentIndex(3)
        pick = QToolButton()
        pick.setText("Select")
        pick.setToolTip("Select every level with n and l up to these")
        pick.clicked.connect(self._quick_pick)
        clear = QToolButton()
        clear.setText("Clear")
        clear.clicked.connect(lambda: self._set_all(False))
        quick.addWidget(QLabel("n ≤"))
        quick.addWidget(self.n_max)
        quick.addWidget(QLabel("l ≤"))
        quick.addWidget(self.l_max)
        quick.addWidget(pick)
        quick.addWidget(clear)
        quick.addStretch(1)
        lv.addLayout(quick)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Level", "Energy (eV)", "Source"])
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        groups: dict = {}
        chosen_set = set(chosen)
        for x in levels:
            g = groups.get(x.l)
            if g is None:
                g = QTreeWidgetItem(self.tree, [f"{'SPDFGHIK'[x.l]} levels"])
                g.setFlags(g.flags() | Qt.ItemIsAutoTristate | Qt.ItemIsUserCheckable)
                groups[x.l] = g
            it = QTreeWidgetItem(g, [x.label, f"{x.binding_eV + IE:.5f}", x.source])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(0, Qt.Checked if x.label in chosen_set else Qt.Unchecked)
            it.setData(0, Qt.UserRole, x.label)
        for l, g in groups.items():
            g.setExpanded(l <= 2)
        self.tree.itemChanged.connect(lambda *_: self._fill_lines())
        lv.addWidget(self.tree, 1)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QLabel("2. Lines between the chosen levels"))
        q2 = QHBoxLayout()
        self.top_n = QSpinBox()
        self.top_n.setRange(1, 500)
        self.top_n.setValue(12)
        b_top = QToolButton()
        b_top.setText("Select strongest")
        b_top.setToolTip("Tick the lines with the largest Einstein A")
        b_top.clicked.connect(self._pick_strongest)
        b_all = QToolButton()
        b_all.setText("All")
        b_all.clicked.connect(lambda: self._tick_lines(True))
        b_none = QToolButton()
        b_none.setText("None")
        b_none.clicked.connect(lambda: self._tick_lines(False))
        q2.addWidget(self.top_n)
        q2.addWidget(b_top)
        q2.addWidget(b_all)
        q2.addWidget(b_none)
        q2.addStretch(1)
        rv.addLayout(q2)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Line", "λ (nm)", "A (s⁻¹)", "f", "f source"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSortingEnabled(True)
        rv.addWidget(self.table, 1)
        self.count = QLabel("")
        rv.addWidget(self.count)
        split.addWidget(right)
        split.setSizes([330, 490])

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self._fill_lines()

    def _items(self):
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            for j in range(g.childCount()):
                yield g.child(j)

    def chosen_levels(self) -> list[str]:
        return [it.data(0, Qt.UserRole) for it in self._items() if it.checkState(0) == Qt.Checked]

    def chosen_lines(self) -> list[str]:
        out = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    def _set_all(self, on: bool) -> None:
        self.tree.blockSignals(True)
        for it in self._items():
            it.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
        self.tree.blockSignals(False)
        self._fill_lines()

    def _quick_pick(self) -> None:
        n, l = self.n_max.value(), self.l_max.currentData()
        by = {x.label: x for x in self.levels}
        self.tree.blockSignals(True)
        for it in self._items():
            x = by[it.data(0, Qt.UserRole)]
            it.setCheckState(0, Qt.Checked if x.n <= n and x.l <= l else Qt.Unchecked)
        self.tree.blockSignals(False)
        self._fill_lines()

    def _fill_lines(self) -> None:
        keep = {self.table.item(r, 0).data(Qt.UserRole) for r in range(self.table.rowCount())
                if self.table.item(r, 0).checkState() == Qt.Checked} if self.table.rowCount() else self._line_keys
        lines = sorted(templates.lines_between(self.all_lines, set(self.chosen_levels())), key=lambda t: -t["A_s"])
        auto = self._auto_lines and not self.table.rowCount()
        strongest = {f"{t['upper_label']}>{t['lower_label']}" for t in lines[:self.top_n.value()]}
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(lines))
        for r, t in enumerate(lines):
            key = f"{t['upper_label']}>{t['lower_label']}"
            it = QTableWidgetItem(f"{t['upper_label']} → {t['lower_label']}")
            it.setData(Qt.UserRole, key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            on = key in strongest if auto else key in keep
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
            self.table.setItem(r, 0, it)
            for c, val in ((1, t["wavelength_nm"]), (2, t["A_s"]), (3, t["fosc"])):
                cell = QTableWidgetItem()
                cell.setData(Qt.DisplayRole, float(f"{val:.4g}"))
                self.table.setItem(r, c, cell)
            self.table.setItem(r, 4, QTableWidgetItem(str(t.get("fosc_source", ""))))
        self.table.setSortingEnabled(True)
        self.table.sortItems(2, Qt.DescendingOrder)          # strongest lines first
        self._auto_lines = False
        self.count.setText(f"{len(lines)} lines between {len(self.chosen_levels())} levels.")

    def _tick_lines(self, on: bool) -> None:
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _pick_strongest(self) -> None:
        rows = sorted(range(self.table.rowCount()), key=lambda r: -float(self.table.item(r, 2).data(Qt.DisplayRole)))
        top = set(rows[: self.top_n.value()])
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setCheckState(Qt.Checked if r in top else Qt.Unchecked)
