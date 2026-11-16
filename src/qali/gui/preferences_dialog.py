"""Edit > Preferences: output folder and ORCA, the theme with a live preview, the extrapolator, and the tables.

Edits stay in the dialog until Apply or OK. Cancel (or Esc) puts back everything as it was when the dialog
opened, including what an earlier Apply saved.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyledItemDelegate,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QDialog,
)

from . import brand, theme
from .extrapolator_settings import ExtrapolatorSettingsPage, reset_button, set_match
from . import units as units_mod
from .advanced_page import AdvancedPage
from .appearance_extras import AppearanceExtras
from .general_page import GeneralPage
from .units_page import UnitsPage
from .tables_page import TablesPage

KEY = "appearance/colors/{theme}/{token}"
QALI_BASE_KEY = "appearance/colors/qali/base"
PAGE_KEY = "preferences/page"
SIZE_KEY = "preferences/size"

# Base colours offered for the Qali theme, besides Custom...
QALI_PRESETS = (("Sage slate", theme.QALI_BASE), ("Ink", "#1f2633"), ("Plum", "#2f2633"),
                ("Clay", "#33291f"), ("Moss", "#24301f"), ("Paper", "#e9e4d8"))
BADGE_ROLE = Qt.UserRole + 1
PAGE_TITLES = {"general": "General", "appearance": "Appearance", "units": "Units and numbers", "advanced": "Advanced", "extrapolator": "Extrapolator", "tables": "Tables and variables"}


def load_overrides(settings) -> None:
    """Read the saved colour choices of every theme into :mod:`qali.gui.theme`."""
    for name in theme.COLOUR_THEMES:
        chosen = {}
        for token, _label in theme.EDITABLE_COLOURS:
            value = settings.value(KEY.format(theme=name, token=token), "")
            if isinstance(value, str) and QColor(value).isValid():
                chosen[token] = value
        theme.set_overrides(name, chosen)
    base = settings.value(QALI_BASE_KEY, "")
    theme.set_qali_base(base if isinstance(base, str) else None)


def save_qali_base(settings, colour: str | None) -> None:
    if colour:
        settings.setValue(QALI_BASE_KEY, colour)
    else:
        settings.remove(QALI_BASE_KEY)


def save_overrides(settings, theme_name: str, colours: dict[str, str]) -> None:
    name = theme.resolve(theme_name)
    for token, _label in theme.EDITABLE_COLOURS:
        key = KEY.format(theme=name, token=token)
        if token in colours:
            settings.setValue(key, colours[token])
        else:
            settings.remove(key)


class _NavDelegate(QStyledItemDelegate):
    """A sidebar row: the page name, and on the right how many of its settings differ from the defaults."""

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        count = index.data(BADGE_ROLE)
        if not count:
            return
        t = theme.current()
        text = str(count)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont(option.font)
        font.setPointSizeF(max(font.pointSizeF() - 1.5, 7.0))
        font.setBold(True)
        painter.setFont(font)
        w = max(18, painter.fontMetrics().horizontalAdvance(text) + 10)
        r = QRectF(option.rect.right() - w - 8, option.rect.center().y() - 9, w, 18)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(t.selection))
        painter.drawRoundedRect(r, 9, 9)
        painter.setPen(QColor(t.accent_ink))
        painter.drawText(r, Qt.AlignCenter, text)
        painter.restore()

    def sizeHint(self, option, index):
        return QSize(option.rect.width(), 34)


class ThemePreview(QWidget):
    """A small Qali window drawn in the colours being chosen, so a choice is seen before it is applied."""

    SPECIES = ("Li", "Na", "K", "Rb", "Cs", "Fr")
    STAGES = (("Rydberg levels", True), ("Transitions", True), ("Lifetimes", True),
              ("Polarizability", False), ("Blackbody", False))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tokens = theme.current()
        self.setMinimumSize(360, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAccessibleName("Preview of the chosen colours")

    def set_tokens(self, tokens) -> None:
        self.tokens = tokens
        self.update()

    def paintEvent(self, _event):  # noqa: C901 - one drawing, top to bottom
        t = self.tokens
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        font = theme.ui_font()
        font.setPointSizeF(8.5)
        p.setFont(font)
        fm = p.fontMetrics()
        frame = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setPen(QPen(QColor(t.border_strong), 1))
        p.setBrush(QColor(t.bg))
        p.drawRoundedRect(frame, 10, 10)
        p.setClipRect(frame.adjusted(1, 1, -1, -1))

        def text(x, y, s, colour, bold=False, rect=None, align=Qt.AlignLeft | Qt.AlignVCenter):
            f = QFont(font)
            f.setBold(bold)
            p.setFont(f)
            p.setPen(QColor(colour))
            if rect is not None:
                p.drawText(rect, align, s)
            else:
                p.drawText(int(x), int(y), s)
            p.setFont(font)

        # title bar: sprig, name, the accent button
        bar = QRectF(frame.left(), frame.top(), frame.width(), 38)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t.surface))
        p.drawRect(bar)
        p.setPen(QPen(QColor(t.border), 1))
        p.drawLine(bar.bottomLeft(), bar.bottomRight())
        p.drawPixmap(QRectF(bar.left() + 10, bar.top() + 9, 20, 20).toRect(),
                     brand.sprig(20, t.text_muted, t.accent))
        text(0, 0, "Qali", t.text, True, QRectF(bar.left() + 36, bar.top(), 80, bar.height()))
        run = QRectF(bar.right() - 92, bar.top() + 8, 82, 22)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t.accent))
        p.drawRoundedRect(run, 5, 5)
        text(0, 0, "Run stage", t.accent_text, True, run, Qt.AlignCenter)

        body_top = bar.bottom() + 1
        console_h = 70
        body = QRectF(frame.left(), body_top, frame.width(), frame.bottom() - body_top - console_h)

        # species list, Na selected (list selection)
        side = QRectF(body.left(), body.top(), 84, body.height())
        p.setPen(QPen(QColor(t.border), 1))
        p.drawLine(side.topRight(), side.bottomRight())
        y = side.top() + 8
        for sp in self.SPECIES:
            row = QRectF(side.left() + 6, y, side.width() - 12, 24)
            if sp == "Na":
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(t.selection))
                p.drawRoundedRect(row, 5, 5)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(brand.species_colour(sp)))
            p.drawEllipse(QRectF(row.left() + 8, row.center().y() - 3.5, 7, 7))
            text(0, 0, sp, t.accent_ink if sp == "Na" else t.text, sp == "Na",
                 QRectF(row.left() + 22, row.top(), row.width() - 22, row.height()))
            y += 26

        # stages with their dots
        main = QRectF(side.right() + 12, body.top() + 8, body.right() - side.right() - 24, body.height() - 16)
        text(0, 0, "Stages · Na", t.text_muted, False, QRectF(main.left(), main.top(), main.width(), 18))
        y = main.top() + 22
        for name, done in self.STAGES:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(t.stage_done if done else t.stage_todo))
            p.drawEllipse(QRectF(main.left() + 2, y + 8, 8, 8))
            text(0, 0, name, t.text, False, QRectF(main.left() + 18, y, main.width() - 18, 24))
            y += 24

        # a small table, one row selected (table selection)
        y += 8
        table = QRectF(main.left(), y, main.width(), 22 * 4)
        p.setPen(QPen(QColor(t.border), 1))
        p.setBrush(QColor(t.surface))
        p.drawRoundedRect(table, 5, 5)
        head = QRectF(table.left(), table.top(), table.width(), 22)
        text(0, 0, "  State            τ (ns)", t.text_muted, False, head)
        rows = (("  3p1/2            16.30", False), ("  3p3/2            16.24", True),
                ("  4s                 38.0", False))
        for i, (label, sel) in enumerate(rows):
            r = QRectF(table.left() + 1, table.top() + 22 * (i + 1), table.width() - 2, 22)
            if sel:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(t.highlight))
                p.drawRect(r)
            text(0, 0, label, t.highlight_text if sel else t.text, False, r)

        # an edit box with selected text (text selection)
        y = table.bottom() + 10
        box = QRectF(main.left(), y, min(main.width(), 220), 26)
        p.setPen(QPen(QColor(t.border_strong), 1))
        p.setBrush(QColor(t.surface))
        p.drawRoundedRect(box, 5, 5)
        lead = "Na  "
        sel = "3p3/2"
        x0 = box.left() + 8
        text(x0, 0, lead, t.text, False, QRectF(x0, box.top(), 40, box.height()))
        sx = x0 + fm.horizontalAdvance(lead)
        sw = fm.horizontalAdvance(sel) + 4
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t.text_select))
        p.drawRect(QRectF(sx - 2, box.top() + 5, sw, box.height() - 10))
        text(0, 0, sel, t.text_select_text, False, QRectF(sx, box.top(), sw, box.height()))

        # an open menu with the hovered item (menu hover)
        menu = QRectF(bar.right() - 150, bar.bottom() + 6, 132, 3 * 24 + 8)
        p.setPen(QPen(QColor(t.border), 1))
        p.setBrush(QColor(t.surface))
        p.drawRoundedRect(menu, 6, 6)
        for i, item in enumerate(("Dark", "Light", "Qali")):
            r = QRectF(menu.left() + 4, menu.top() + 4 + 24 * i, menu.width() - 8, 24)
            if i == 2:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(t.menu_select))
                p.drawRoundedRect(r, 4, 4)
            text(0, 0, "   " + item, t.text, False, r)

        # console with a warning line and a delete button (warning and delete)
        con = QRectF(frame.left(), frame.bottom() - console_h, frame.width(), console_h)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t.surface))
        p.drawRect(con)
        p.setPen(QPen(QColor(t.border), 1))
        p.drawLine(con.topLeft(), con.topRight())
        mono = theme.mono_font()
        mono.setPointSizeF(8)
        p.setFont(mono)
        lines = (("$ qali --pipeline lifetimes Na", t.text_muted), ("Lifetimes written: 282 states", t.text),
                 ("2 transitions dropped", t.danger))
        for i, (line, colour) in enumerate(lines):
            p.setPen(QColor(colour))
            p.drawText(QRectF(con.left() + 12, con.top() + 6 + 19 * i, con.width() - 110, 19),
                       Qt.AlignLeft | Qt.AlignVCenter, line)
        p.setFont(font)
        delete = QRectF(con.right() - 82, con.bottom() - 32, 70, 24)
        p.setPen(QPen(QColor(t.border), 1))
        p.setBrush(QColor(t.raised))
        p.drawRoundedRect(delete, 5, 5)
        text(0, 0, "Delete", t.danger, False, delete, Qt.AlignCenter)
        p.end()


def _theme_icon(name: str) -> QPixmap:
    """A small window swatch of theme *name* for its card."""
    pm = QPixmap(56, 40)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    rect = QRectF(1, 1, 26, 18)
    if name == theme.SYSTEM:
        dark, light = theme.base_tokens(theme.DARK), theme.base_tokens(theme.LIGHT)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(dark.bg))
        p.drawRoundedRect(rect, 4, 4)
        p.setBrush(QColor(light.bg))
        p.drawRoundedRect(QRectF(14, 1, 13, 18), 4, 4)
        p.drawRect(QRectF(14, 1, 4, 18))
        accent, border = dark.accent, dark.border_strong
    else:
        t = theme.base_tokens(name)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t.bg))
        p.drawRoundedRect(rect, 4, 4)
        p.setBrush(QColor(t.surface))
        p.drawRect(QRectF(3, 3, 7, 14))
        accent, border = t.accent, t.border_strong
    p.setBrush(QColor(accent))
    p.drawRoundedRect(QRectF(14, 12, 10, 4), 2, 2)
    p.setPen(QPen(QColor(border), 1))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(rect, 4, 4)
    p.end()
    return pm


class PreferencesDialog(QDialog):
    """General, Appearance (with a live preview), Extrapolator and Tables and variables, with a search over all of them
    and, beside each page's name, how many of its settings differ from the defaults."""

    TABS = ("general", "appearance", "units", "extrapolator", "tables", "advanced")

    def __init__(self, parent, theme_name: str, settings, apply_callback, root=None, species=None, start_tab=None,
                 set_theme=None, default_root=None, set_root=None, units_changed=None, look_changed=None):
        super().__init__(parent)
        self.setObjectName("preferences")
        self.setWindowTitle("Preferences")
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setSizeGripEnabled(True)
        self.settings, self.apply_callback = settings, apply_callback
        self.set_theme = set_theme
        self._theme_original = theme_name
        self._theme_applied = theme_name
        self.theme_choice = theme_name
        self._original = {n: theme.overrides_for(n) for n in theme.COLOUR_THEMES}
        self.chosen = {n: dict(c) for n, c in self._original.items()}
        self._base_original = theme._qali_base
        self.base_choice = self._base_original     # the Qali theme's base colour; None is the default
        self._closing = False
        size = settings.value(SIZE_KEY)
        self.resize(size if isinstance(size, QSize) and size.isValid() else QSize(1080, 760))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        outer.addLayout(body, 1)
        self.nav = QListWidget()                 # pages down the left side
        self.nav.setObjectName("prefsNav")
        self.nav.setFixedWidth(210)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav.setItemDelegate(_NavDelegate(self.nav))
        self.nav.setSpacing(1)
        self.pages = QStackedWidget()
        body.addWidget(self.nav)
        body.addWidget(self.pages, 1)
        self.nav.currentRowChanged.connect(self._page_changed)

        self.swatches: dict[tuple[str, str], QPushButton] = {}
        self.look_changed = look_changed
        self.extras = AppearanceExtras(settings)
        self.extras.changed.connect(self._extras_edited)
        self.general = None
        if root is not None:
            if default_root is None:
                from qali.core.workspace import default_workspace
                default_root = default_workspace().root
            self.general = GeneralPage(settings, root, default_root, set_root)
            self._add_page(self._titled(self.general, "General"), "general")
            self.general.changed.connect(self._schedule_counts)
        self._add_page(self._build_appearance(), "appearance")
        self.units_changed = units_changed
        self.units = UnitsPage(settings)
        self._add_page(self._titled(self.units, "Units and numbers"), "units")
        self.units.changed.connect(self._schedule_counts)
        self.extrapolator = None
        self.tables = None
        if root is not None:
            self.extrapolator = ExtrapolatorSettingsPage(root, species)
            self._add_page(self._titled(self.extrapolator, "Extrapolator",
                                        "Fit settings, and the model behind the extrapolated f-values."), "extrapolator")
            self.tables = TablesPage(root)
            self._add_page(self._titled(self.tables, "Tables and variables",
                                        "Qali's built-in numbers and data files."), "tables")
            self.extrapolator.changed.connect(self._schedule_counts)
            self.tables.changed.connect(self._schedule_counts)
        self.advanced = AdvancedPage(settings)
        self._add_page(self._titled(self.advanced, "Advanced"), "advanced")
        self.advanced.reset_requested.connect(self._reset_everything)
        self.advanced.imported.connect(self._imported)
        outer.addWidget(self._build_footer())

        start = start_tab if start_tab in self.TABS else settings.value(PAGE_KEY, "general")
        names = [self.nav.item(i).data(Qt.UserRole) for i in range(self.nav.count())]
        self.nav.setCurrentRow(names.index(start) if start in names else 0)
        self._counts_timer = QTimer(self)
        self._counts_timer.setSingleShot(True)
        self._counts_timer.setInterval(150)
        self._counts_timer.timeout.connect(self._update_counts)
        self._refresh()
        self._update_counts()
        self.nav.setFocus()

    # ---- frame: header with the search, sidebar, footer ---------------------------------------------------
    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("prefsHeader")
        header.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(header)
        lay.setContentsMargins(18, 10, 18, 10)
        lay.setSpacing(10)
        t = theme.current()
        mark = QLabel()
        mark.setPixmap(brand.sprig(22, t.text_muted, t.accent))
        lay.addWidget(mark)
        title = QLabel("Preferences")
        title.setObjectName("prefsTitle")
        lay.addWidget(title)
        slash = QLabel("/")
        slash.setObjectName("prefsCrumb")
        lay.addWidget(slash)
        self.crumb = QLabel()
        self.crumb.setObjectName("prefsCrumb")
        lay.addWidget(self.crumb)
        lay.addStretch(1)
        self.search = QLineEdit()
        self.search.setObjectName("prefsSearch")
        self.search.setPlaceholderText("Search all settings  (Ctrl+F)")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(300)
        self.search.textChanged.connect(self._search)
        self.search.returnPressed.connect(self._search_next)
        lay.addWidget(self.search)
        QShortcut(QKeySequence.StandardKey.Find, self, activated=lambda: (self.search.setFocus(),
                                                                          self.search.selectAll()))
        return header

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        footer.setObjectName("prefsFooter")
        footer.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(footer)
        lay.setContentsMargins(16, 10, 16, 10)
        self.defaults_btn = QPushButton("Defaults (this page)")
        self.defaults_btn.clicked.connect(self._reset_all)
        lay.addWidget(self.defaults_btn)
        lay.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self._cancel)
        apply = QPushButton("Apply")
        apply.clicked.connect(self._apply)
        ok = QPushButton("OK")
        ok.setObjectName("primary")
        ok.setDefault(True)
        ok.clicked.connect(self._ok)
        for b in (cancel, apply, ok):
            lay.addWidget(b)
        return footer

    def _add_page(self, widget, key: str):
        self.pages.addWidget(widget)
        item = QListWidgetItem(PAGE_TITLES[key])
        item.setData(Qt.UserRole, key)
        item.setSizeHint(QSize(0, 34))
        self.nav.addItem(item)

    @staticmethod
    def _titled(widget: QWidget, title: str, subtitle: str = "") -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(26, 20, 22, 12)
        lay.setSpacing(4)
        head = QLabel(title)
        head.setObjectName("pageTitle")
        lay.addWidget(head)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("muted")
            lay.addWidget(sub)
        lay.addSpacing(10)
        lay.addWidget(widget, 1)
        return page

    def _page_changed(self, row: int) -> None:
        if row < 0:
            return
        self.pages.setCurrentIndex(row)
        self.crumb.setText(self.nav.item(row).text())
        self.defaults_btn.setVisible(self.nav.item(row).data(Qt.UserRole) != "advanced")

    def _schedule_counts(self) -> None:
        self._counts_timer.start()

    def _update_counts(self) -> None:
        """Beside each page: how many of its settings differ from the defaults."""
        counts = {"appearance": sum(len(c) for c in self.chosen.values()) + (1 if self.base_choice else 0)
                  + self.extras.changed_count()}
        if self.general is not None:
            counts["general"] = self.general.changed_count()
        counts["units"] = self.units.changed_count()
        if self.extrapolator is not None:
            counts["extrapolator"] = self.extrapolator.changed_count()
        if self.tables is not None:
            counts["tables"] = self.tables.changed_count()
        for i in range(self.nav.count()):
            item = self.nav.item(i)
            item.setData(BADGE_ROLE, counts.get(item.data(Qt.UserRole), 0))

    # ---- Appearance -------------------------------------------------------------------------------------
    def _build_appearance(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(26, 20, 22, 16)
        row.setSpacing(22)

        left = QWidget()
        left.setFixedWidth(400)
        lay = QVBoxLayout(left)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        head = QLabel("Appearance")
        head.setObjectName("pageTitle")
        sub = QLabel("Changes show in the preview before you apply them.")
        sub.setObjectName("muted")
        heads = QVBoxLayout()
        heads.setSpacing(4)
        heads.addWidget(head)
        heads.addWidget(sub)
        lay.addLayout(heads)

        self.theme_title = QLabel("Theme")
        self.theme_title.setObjectName("cardTitle")
        lay.addWidget(self.theme_title)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.theme_group = QButtonGroup(self)
        self.theme_group.setExclusive(True)
        self.theme_cards: dict[str, QToolButton] = {}
        for i, key in enumerate(theme.THEME_NAMES):
            card = QToolButton()
            card.setObjectName("themeCard")
            card.setCheckable(True)
            card.setText(theme.THEME_LABELS[key])
            card.setIcon(_theme_icon(key))
            card.setIconSize(QSize(28, 20))
            card.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            card.setMinimumHeight(44)
            card.clicked.connect(lambda _c=False, k=key: self._choose_theme(k))
            self.theme_group.addButton(card)
            self.theme_cards[key] = card
            grid.addWidget(card, i // 2, i % 2)
        lay.addLayout(grid)
        self.theme_cards.get(self.theme_choice, self.theme_cards[theme.DARK]).setChecked(True)

        self.base_box = QWidget()
        blay = QVBoxLayout(self.base_box)
        blay.setContentsMargins(0, 0, 0, 0)
        blay.setSpacing(8)
        self.base_title = QLabel("Qali base colour")
        self.base_title.setObjectName("cardTitle")
        blay.addWidget(self.base_title)
        brow = QHBoxLayout()
        brow.setSpacing(8)
        self.base_buttons: dict[str, QToolButton] = {}
        for name, colour in QALI_PRESETS:
            b = QToolButton()
            b.setFixedSize(30, 30)
            b.setToolTip(name)
            b.setAccessibleName(name)
            b.clicked.connect(lambda _c=False, c=colour: self._set_base(c))
            self.base_buttons[colour] = b
            brow.addWidget(b)
        self.base_btn = QPushButton("Custom...")
        self.base_btn.clicked.connect(self._pick_base)
        brow.addWidget(self.base_btn)
        brow.addStretch(1)
        blay.addLayout(brow)
        lay.addWidget(self.base_box)

        card = QFrame()
        card.setObjectName("prefsCard")
        self.colour_layout = QVBoxLayout(card)
        self.colour_layout.setContentsMargins(16, 10, 12, 10)
        self.colour_layout.setSpacing(0)
        self.colours_title = QLabel()
        self.colours_title.setObjectName("cardTitle")
        self.colour_layout.addWidget(self.colours_title)
        self.colour_layout.addSpacing(6)
        self.colour_rows = QVBoxLayout()
        self.colour_rows.setSpacing(0)
        self.colour_layout.addLayout(self.colour_rows)
        lay.addWidget(card)
        lay.addWidget(self.extras)
        lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(left)
        scroll.setFixedWidth(414)
        row.addWidget(scroll)

        right = QVBoxLayout()
        right.setSpacing(8)
        cap = QHBoxLayout()
        cap_l = QLabel("Preview")
        cap_l.setObjectName("cardTitle")
        self.preview_caption = QLabel()
        self.preview_caption.setObjectName("muted")
        cap.addWidget(cap_l)
        cap.addStretch(1)
        cap.addWidget(self.preview_caption)
        right.addLayout(cap)
        self.preview = ThemePreview()
        right.addWidget(self.preview, 1)
        row.addLayout(right, 1)
        self._build_colour_rows()
        return page

    @property
    def shown_theme(self) -> str:
        """The concrete theme whose colours the page shows (Follow system resolves to Dark or Light)."""
        return theme.resolve(self.theme_choice)

    def _build_colour_rows(self) -> None:
        """One row per editable colour of the theme shown: name, 'changed', swatch, reset icon."""
        while self.colour_rows.count():
            widget = self.colour_rows.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        self.swatches = {}
        self._resets: dict[str, QToolButton] = {}
        self._marks: dict[str, QLabel] = {}
        self.colour_labels: dict[str, QLabel] = {}
        name = self.shown_theme
        for token, label in theme.EDITABLE_COLOURS:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 4, 0, 4)
            h.setSpacing(8)
            text = QLabel(label)
            h.addWidget(text)
            h.addStretch(1)
            mark = QLabel("changed")
            mark.setObjectName("changedMark")
            h.addWidget(mark)
            swatch = QPushButton()
            swatch.setFixedSize(QSize(92, 26))
            swatch.setAccessibleName(f"{label} colour")
            swatch.clicked.connect(lambda _c=False, n=name, t=token: self._pick(n, t))
            h.addWidget(swatch)
            reset = reset_button(lambda _c=False, n=name, t=token: self._reset(n, t))
            reset.setFixedWidth(26)
            keep = reset.sizePolicy()
            keep.setRetainSizeWhenHidden(True)
            reset.setSizePolicy(keep)
            h.addWidget(reset)
            self.colour_rows.addWidget(row)
            self.swatches[(name, token)] = swatch
            self._resets[token], self._marks[token], self.colour_labels[token] = reset, mark, text
        self.colours_title.setText(f"{theme.THEME_LABELS[name]} theme colours")

    def _choose_theme(self, key: str) -> None:
        rebuild = theme.resolve(key) != self.shown_theme
        self.theme_choice = key
        if key in self.theme_cards:
            self.theme_cards[key].setChecked(True)
        if rebuild:
            self._build_colour_rows()
            if self.search.text().strip():
                self._search(self.search.text())
        self._refresh()

    def _current(self, name: str, token: str) -> str:
        return self.chosen[name].get(token) or getattr(theme.base_tokens(name, self.base_choice), token)

    def _refresh(self):
        name = self.shown_theme
        for (n, token), button in self.swatches.items():
            colour = self._current(n, token)
            fg = "#000000" if QColor(colour).lightness() > 140 else "#ffffff"
            button.setText(colour.upper())
            button.setStyleSheet(f"QPushButton {{ background: {colour}; color: {fg}; border: 1px solid #888;"
                                 f" border-radius: 5px; font-family: Consolas, 'JetBrains Mono', monospace; }}")
            changed = token in self.chosen[n]
            self._resets[token].setVisible(changed)
            self._marks[token].setVisible(changed)
        base = self.base_choice or theme.QALI_BASE
        for colour, b in self.base_buttons.items():
            on = colour.lower() == base.lower()
            ring = theme.current().accent_ink if on else "#888888"
            b.setStyleSheet(f"QToolButton {{ background: {colour}; border: {2 if on else 1}px solid {ring};"
                            f" border-radius: 15px; }}")
        custom = base.lower() not in {c.lower() for _n, c in QALI_PRESETS}
        self.base_btn.setText(f"Custom  {base.upper()}" if custom else "Custom...")
        self.base_box.setVisible(name == theme.QALI)
        self.preview.set_tokens(theme.tokens_with(name, self.chosen[name], self.base_choice))
        self.preview_caption.setText(f"{theme.THEME_LABELS[self.theme_choice]} theme"
                                     + (f" ({theme.THEME_LABELS[name]})" if self.theme_choice == theme.SYSTEM else ""))
        if hasattr(self, "_counts_timer"):
            self._schedule_counts()

    def _set_base(self, colour: str | None) -> None:
        self.base_choice = None if not colour or colour.lower() == theme.QALI_BASE else QColor(colour).name()
        self._refresh()

    def _pick_base(self):
        colour = QColorDialog.getColor(QColor(self.base_choice or theme.QALI_BASE), self, "Choose colour")
        if colour.isValid():
            self._set_base(colour.name())

    def _reset_base(self):
        self._set_base(None)

    def _pick(self, name: str, token: str):
        colour = QColorDialog.getColor(QColor(self._current(name, token)), self, "Choose colour")
        if colour.isValid():
            self.chosen[name][token] = colour.name()
            self._refresh()

    def _reset(self, name: str, token: str):
        self.chosen[name].pop(token, None)
        self._refresh()

    def _appearance_labels(self) -> list[QLabel]:
        return [self.theme_title, self.base_title, *self.colour_labels.values()]

    def _appearance_texts(self) -> list[str]:
        return ([lbl.text() for lbl in self._appearance_labels()]
                + [theme.THEME_LABELS[k] for k in theme.THEME_NAMES] + ["Preview", "Colours"])

    # ---- search -----------------------------------------------------------------------------------------
    def _search(self, text: str) -> None:
        """Hide the pages without a match, mark the matching settings, and move off a page with none."""
        needle = text.strip().lower()
        hits = {"appearance": 0}
        for label in self._appearance_labels():
            on = bool(needle) and needle in label.text().lower()
            set_match(label, on)
        if needle:
            hits["appearance"] = sum(needle in t.lower() for t in self._appearance_texts())
        hits["appearance"] += self.extras.highlight(text)
        if self.general is not None:
            hits["general"] = self.general.highlight(text)
        hits["units"] = self.units.highlight(text)
        hits["advanced"] = self.advanced.highlight(text)
        if self.extrapolator is not None:
            hits["extrapolator"] = self.extrapolator.highlight(text)
        if self.tables is not None:
            hits["tables"] = self.tables.highlight(text)
        first = None
        for i in range(self.nav.count()):
            item = self.nav.item(i)
            hidden = bool(needle) and not hits.get(item.data(Qt.UserRole))
            item.setHidden(hidden)
            if not hidden and first is None:
                first = i
        current = self.nav.currentItem()
        if current is not None and current.isHidden() and first is not None:
            self.nav.setCurrentRow(first)
        if needle and first is None:
            self.crumb.setText("No settings match")
        elif self.nav.currentItem() is not None:
            self.crumb.setText(self.nav.currentItem().text())

    def _search_next(self) -> None:
        """Enter in the search box: go to the next page that has a match."""
        visible = [i for i in range(self.nav.count()) if not self.nav.item(i).isHidden()]
        if len(visible) > 1:
            later = [i for i in visible if i > self.nav.currentRow()]
            self.nav.setCurrentRow((later or visible)[0])

    # ---- buttons ----------------------------------------------------------------------------------------
    def _reset_all(self):
        page = self.nav.currentItem().data(Qt.UserRole) if self.nav.currentItem() else "appearance"
        what = {"general": "the output folder, the ORCA program and the start-up options",
                "appearance": "the theme colours, the Qali base colour, text sizes and layout",
                "units": "the units and number format",
                "extrapolator": "the fit settings and every species' model parameters",
                "tables": "the table or file on screen"}[page]
        if QMessageBox.question(self, "Defaults", f"Restore the defaults of {what}?",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._restore_page_defaults(page)

    def _restore_page_defaults(self, page: str) -> None:
        if page == "general" and self.general is not None:
            self.general.reset_all()
        elif page == "units":
            self.units.reset_all()
        elif page == "extrapolator" and self.extrapolator is not None:
            self.extrapolator.reset_all()
        elif page == "tables" and self.tables is not None:
            self.tables.reset_all()
        else:
            for colours in self.chosen.values():
                colours.clear()
            self.base_choice = None
            self.extras.reset_all()
            self._refresh()
        self._schedule_counts()

    def _reset_everything(self) -> None:
        """Advanced > Reset all preferences: every page back to its defaults; OK keeps it, Cancel does not."""
        for page in (self.general, self.units, self.extrapolator):
            if page is not None:
                page.reset_all()
        if self.tables is not None:
            self.tables.reset_everything()
        for colours in self.chosen.values():
            colours.clear()
        self.base_choice = None
        self.extras.reset_all()
        self._choose_theme(theme.DARK)
        self._update_counts()

    def _imported(self) -> None:
        """Advanced > Import: the store holds the file's preferences; put them in force and close."""
        load_overrides(self.settings)
        units_mod.load(self.settings)
        brand.load_colours(self.settings)
        theme.load_fonts(self.settings)
        name = self.settings.value("appearance/theme", theme.DARK)
        if self.set_theme is not None and name in theme.THEME_NAMES:
            self.set_theme(name)
        self.apply_callback()
        if self.look_changed is not None:
            self.look_changed()
        if self.units_changed is not None:
            self.units_changed()
        self._remember()
        self._closing = True
        self.accept()

    def _extras_edited(self) -> None:
        self._schedule_counts()

    def _apply(self):
        if self.set_theme is not None and self.theme_choice != self._theme_applied:
            self._theme_applied = self.theme_choice
            self.set_theme(self.theme_choice)
        for name, colours in self.chosen.items():
            theme.set_overrides(name, colours)
        theme.set_qali_base(self.base_choice)
        look = self.extras.is_different()
        if look:
            self.extras.apply()
        self.apply_callback()
        if look and self.look_changed is not None:
            self.look_changed()
        before = units_mod.current()
        self.units.apply()
        if units_mod.current() != before and self.units_changed is not None:
            self.units_changed()
        if self.extrapolator is not None:
            left_out = self.extrapolator.apply()
            if left_out:
                QMessageBox.warning(self, "Extrapolator", "Not saved:\n- " + "\n- ".join(left_out))
        if self.tables is not None:
            left_out = self.tables.apply()
            if left_out:
                QMessageBox.warning(self, "Tables and variables", "Not saved:\n- " + "\n- ".join(left_out))
        if self.general is not None:            # last: a new folder must not take the other pages' files
            left_out = self.general.apply()
            if left_out:
                QMessageBox.warning(self, "General", "Not saved:\n- " + "\n- ".join(left_out))
        self._refresh()
        self._update_counts()

    def _remember(self) -> None:
        item = self.nav.currentItem()
        if item is not None:
            self.settings.setValue(PAGE_KEY, item.data(Qt.UserRole))
        self.settings.setValue(SIZE_KEY, self.size())

    def _ok(self):
        self._apply()
        for name, colours in self.chosen.items():
            save_overrides(self.settings, name, colours)
        save_qali_base(self.settings, self.base_choice)
        self._remember()
        self._closing = True
        self.accept()

    def _cancel(self):
        self.reject()

    def reject(self):
        """Cancel, Esc and the close button: put back everything as it was when the dialog opened."""
        if not self._closing:
            self._closing = True
            if self.set_theme is not None and self._theme_applied != self._theme_original:
                self.set_theme(self._theme_original)
            for name, colours in self._original.items():
                theme.set_overrides(name, colours)
            theme.set_qali_base(self._base_original)
            undo_look = self.extras.applied
            if undo_look:
                self.extras.undo_apply()
            self.apply_callback()
            if undo_look and self.look_changed is not None:
                self.look_changed()
            for page in (self.extrapolator, self.tables, self.general):
                if page is not None and page.applied:
                    page.undo_apply()
            if self.units.applied:
                self.units.undo_apply()
                if self.units_changed is not None:
                    self.units_changed()
            self._remember()
        super().reject()
