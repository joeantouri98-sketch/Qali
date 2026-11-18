"""Help and pipeline reference dialog for the Qali desktop app."""

from __future__ import annotations

from html import escape
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
)

from .project_io import open_pipeline_overview_html


class HelpDialog(QDialog):
    """Show generated CLI reference, stage explanations, and usage notes."""

    def __init__(self, root: Path, pipeline, stage_physics, parent=None):
        super().__init__(parent)
        self.root = root
        self.pipeline = pipeline
        self.stage_physics = stage_physics
        self.setWindowTitle("Qali Control Panel - Help")
        self.resize(800, 640)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        flags_view = QTextBrowser()
        flags_view.setHtml(self._flags_html())
        tabs.addTab(flags_view, "CLI flags reference")

        stages_view = QTextBrowser()
        stages_view.setHtml(self._stages_html())
        tabs.addTab(stages_view, "Pipeline stages")

        about_view = QTextBrowser()
        about_view.setHtml(self._about_html())
        tabs.addTab(about_view, "About / tips")

        button_row = QHBoxLayout()
        overview_button = QPushButton("Open full Pipeline Overview (HTML)...")
        overview_button.clicked.connect(
            lambda: open_pipeline_overview_html(self.root, self)
        )
        button_row.addWidget(overview_button)
        button_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

    def _flags_html(self):
        parts = [
            "<h2>CLI flags, by pipeline stage</h2>",
            "<p>Every field in a stage's options form maps to one command-line "
            "flag of that pipeline. Leaving a field blank simply omits the flag, "
            "so the pipeline's own default applies.</p>",
        ]
        for stage in self.pipeline:
            parts.append(
                f"<h3>{escape(stage['label'])} &nbsp; "
                f"<code>qali --pipeline {escape(stage['pipeline'])}</code></h3>"
            )
            parts.append(f"<p>{escape(stage['desc'])}</p>")
            flags = stage.get("flags", [])
            if not flags:
                parts.append("<p><i>No configurable options for this stage.</i></p>")
                continue
            parts.append(
                "<table cellspacing='0' cellpadding='5' border='1' width='100%' "
                "style='border-collapse:collapse;'>"
                "<tr><th align='left'>Flag</th><th align='left'>Field</th>"
                "<th align='left'>Default</th><th align='left'>Meaning</th></tr>"
            )
            for spec in flags:
                kind = spec["kind"]
                default = spec.get("default", "")
                if callable(default):
                    default = default(self.root)
                if kind == "flag":
                    default_display = "off"
                elif kind == "invflag":
                    default_display = "on"
                elif kind == "choice":
                    default_display = escape(str(default))
                    for choice in spec["choices"]:
                        if isinstance(choice, tuple) and choice[0] == default:
                            default_display = escape(choice[1])
                            break
                else:
                    default_display = escape(str(default)) if default else "<i>(blank)</i>"
                help_text = spec.get("help") or spec["label"]
                parts.append(
                    f"<tr><td><code>{escape(spec['name'])}</code></td>"
                    f"<td>{escape(spec['label'])}</td><td>{default_display}</td>"
                    f"<td>{escape(help_text)}</td></tr>"
                )
            parts.append("</table>")
        return "".join(parts)

    def _stages_html(self):
        parts = [
            "<h2>What each stage computes</h2>",
            "<p>Short physics context for each pipeline stage; see the full "
            "Pipeline Overview document (button below) for full derivations.</p>",
        ]
        for stage in self.pipeline:
            parts.append(
                f"<h3>{escape(stage['label'])}</h3>"
                f"<p>{escape(self.stage_physics.get(stage['id'], ''))}</p>"
            )
        return "".join(parts)

    @staticmethod
    def _about_html():
        return """
        <h2>About this panel</h2>
        <p>This is a control panel over the Qali pipelines. It runs each
        stage as a <code>qali --pipeline</code> subprocess and displays the
        JSON, CSV, and HTML files it produces. It does not reimplement the
        physics. Qali is free software under the GNU GPL v3 or later.</p>
        <h3>Output folder</h3>
        <p>Results go to Qali's per-user data folder unless another folder is
        chosen with <i>File &gt; Open output folder...</i> (<i>Use default
        output folder</i> goes back). The bundled data and example results
        are copied into the folder in use; existing files are never
        overwritten, so after an update re-run a stage to refresh its plot
        or delete the old file first. A Qali source checkout keeps its
        outputs under <code>.qali/</code>.</p>
        <h3>Species IDs and new species</h3>
        <p>Bare element symbols such as <code>Na</code> and <code>Cs</code>
        represent neutral atoms. Ion IDs such as <code>Na_c1</code>,
        <code>Mg_c1</code>, and <code>Na_a1</code> are accepted when registered.
        Choosing an atom with no bundled data (for example Mg) opens a dialog
        to import its NIST levels and lines and type its numbers.
        <i>Edit &gt; Species properties...</i> shows one element's built-in
        values and data files and flags what is missing.</p>
        <h3>Grotrian diagram</h3>
        <p>The stage needs ORCA data (TD-DFT or SOC) for the species; run the
        ORCA stage first. The page shows levels in columns by orbital l, with
        fine-structure bars where J is known. Use the panel on the left to
        filter by spectral region, level class, f-value source and grade, n,
        wavelength, minimum f, and ground or excited lower level; click a
        section title to fold it. Click a line or table row to select it,
        shift-click to add, double-click to isolate, right-click for
        <i>View more</i> (every value for that transition, including the
        extrapolation model f with its 68% and 95% bands). Click a level or
        an ORCA marker to pin it. <i>Export</i> saves the diagram as SVG, PNG
        or PDF on a light background, or the matching transitions as CSV.
        ORCA markers are drawn at their exact energy, in the column of the
        nearest NIST multiplet.</p>
        <h3>File properties and versions</h3>
        <p>Every output a stage writes records the Qali version and stage
        that made it. Right-click a file in the <i>Plots</i>, <i>Data</i> or
        <i>Shared</i> list and choose <i>Properties...</i> to see it with the
        file's dates. &#9888; marks an output that is out of date: its stage
        has changed since, or a stage it was built from was rerun. An output
        made before these records existed says so in its tooltip; rerun its
        stage to be sure it is current.</p>
        <h3>Figures</h3>
        <p>The <i>Figures</i> tab (in the viewer and the bottom panel) draws
        publication figures with matplotlib from the current results: Grotrian
        diagram, quantum defects, scaling laws, theory vs experiment,
        &alpha;(&lambda;), &alpha;<sub>0</sub> and C<sub>6</sub> by line
        (with convergence and uncertainty budget), decay branching and
        cascades,
        df/d&epsilon; across the ionization limit, blackbody shifts and
        rates, Breit&ndash;Rabi diagrams, Feshbach a(B), radial
        wavefunctions, spectra (absorption, emission, D lines), trap depth and
        scattering against wavelength, and a plot builder for any table in the
        stage outputs or any JSON file. The builder can put a series on a
        right-hand axis or in an inset, draw heat maps and contours of a z
        column, and break an axis. For a Grotrian diagram, <i>Levels and lines...</i> chooses
        the levels and then the lines between them. <i>Input</i> picks which
        result each figure reads: the usual output, a run saved under a name
        (<i>Save as</i>), or another file of that stage.</p>
        <p>On the figure, the mouse wheel zooms (Shift: x only, Alt: y only),
        dragging pans, right-dragging zooms to a box and a double-click shows
        the full view. The ranges appear under <i>Axes</i>, and the export,
        script and recipe use the same view. Ctrl + wheel magnifies the whole
        figure instead (Ctrl + 0 fits it).</p>
        <h3>Plot windows</h3>
        <p>The &#8599; button beside a panel's tabs opens its plot in a window of
        its own; each press opens another. Ctrl + wheel magnifies the whole plot
        in the viewer and in these windows.</p>
        <p>Size and style follow a journal (APS, Nature, thesis, slides) or a
        custom size. <i>Export</i> saves PDF, SVG, EPS, PNG or TIFF;
        <i>Script</i> writes a Python script that redraws the figure;
        <i>Data</i> saves the plotted numbers as CSV; <i>Save</i> keeps a
        recipe (<code>.qfig</code>) that <code>qali --pipeline figure</code>
        redraws after a stage changes. LaTeX text needs <code>latex</code>,
        <code>dvipng</code> and the cm-super fonts. <i>Edit &gt; Preferences
        &gt; Plots</i> sets what new figures start with.</p>
        <h3>Extrapolated f-values</h3>
        <p>For the supported alkalis the Lifetimes stage can extrapolate the
        ns&rarr;np oscillator strengths to high n (<code>--extrapolation</code>,
        set up in <i>Edit &gt; Preferences &gt; Extrapolator</i>). Ground
        ns&rarr;np lines with no measured, published or NIST-series value take
        the calibrated model (core-polarized dipole) in every stage, not QDT
        Numerov; with <i>Off</i> that is the model alone. <i>Show</i> also
        records the anchored fit next to the measured values; <i>Use</i> fills
        the gaps with the fit, tagged <code>extrapolated</code>, and the
        polarizability and blackbody stages follow. Run the later stages with
        the same setting.</p>
        <h3>Tables and variables</h3>
        <p><i>Edit &gt; Preferences &gt; Tables and variables</i> lists the
        built-in constants and data files. Edits are kept in the
        <code>tables/</code> folder of the output folder and never replace
        the bundled values; Restore Defaults brings them back. Add... and
        Remove work on the open lists.</p>
        <p><i>Published matrix elements</i> lists published J-resolved
        matrix elements |D| next to NIST's value for the same line. A ticked
        line uses the published value in both Polarizability and Lifetimes;
        an unticked one uses NIST's, or the literature average where NIST has
        none. All start ticked. Add... takes a line of your own.</p>
        <h3>Save as</h3>
        <p>The <i>Save as</i> field under a stage's output options files that
        run's results under a name, so another run with different settings
        does not overwrite them.</p>
        <h3>Offline plot rendering</h3>
        <p>Plots render offline with the bundled plotly.js. The GUI never
        modifies the report files in the output folder. Exports from the
        viewer ask where to save.</p>
        <h3>ORCA stages</h3>
        <p>The ORCA stages require ORCA to be installed and callable as
        <code>orca</code> on the system PATH, or named by the
        <code>ORCA_EXE</code> environment variable. SOC and NEVPT2 calculations can
        take a long time. Use Stop if a running subprocess must be cancelled.</p>
        <h3>Console, find and tests</h3>
        <p>The command preview shows the command that will be executed. The
        Console tab shows stdout and stderr from the running subprocess.
        Ctrl+F opens a find bar in the JSON viewer, the data editor and the
        tables page. <i>Run &gt; Tests...</i> runs the test suite with
        progress and details.</p>
        <h3>Custom dataset editor</h3>
        <p>The Edit menu provides editors for JSON and CSV datasets. JSON is
        validated before saving, and CSV headers and row widths are checked.</p>
        <h3>Licence and credits</h3>
        <p>GPL-3.0-or-later. Plotly.js (MIT) is bundled for the offline plots.
        ARC (BSD-3-Clause) is an optional dependency. ORCA is separate software
        with its own licence. Cite Qali and your data sources (NIST ASD and
        others); see <code>REFERENCES.md</code>.</p>
        """
