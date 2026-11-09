"""Model library dialog: browse and install open-source models from OpenModelDB inside MUSS."""
import json
import threading
import time

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
)

USES = {
    "All uses": None,
    "Live action / video": {"video-frame", "photo", "general-upscaler"},
    "Anime / cartoon": {"anime", "cartoon"},
    "Restoration / compression": {"restoration", "compression-removal", "jpeg", "denoise", "deblur"},
}
NON_COMMERCIAL = ("NC",)


class _Bridge(QObject):
    loaded = Signal(object, object)          # (models, error)
    progress = Signal(int, int)
    installed = Signal(object, object)       # (spec, error)


class ModelLibrary(QDialog):
    """on_installed(spec) is called after a model is installed or updated."""

    def __init__(self, parent, settings, on_installed):
        super().__init__(parent)
        self.setWindowTitle("Model library - OpenModelDB")
        self.resize(1180, 720)
        self.settings = settings
        self.on_installed = on_installed
        self.models = []
        self.shown = []
        self.busy = False
        self.cancel_flag = False
        self.b = _Bridge()
        self.b.loaded.connect(self._loaded)
        self.b.progress.connect(self._progress)
        self.b.installed.connect(self._installed)
        try:
            self.seen = set(json.loads(settings.value("catalog_seen", "[]")))
        except (TypeError, ValueError):
            self.seen = set()
        self._build()
        self._load(force=False)

    # ------------------------------------------------------------------ layout
    def _build(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search name, author, tags, description...")
        self.scale_box = QComboBox()
        self.scale_box.addItems(["Any scale", "2×  (best for 1080p → 4K)", "4×", "1×  (restoration)", "Other"])
        self.scale_box.setCurrentIndex(1)
        self.use_box = QComboBox()
        self.use_box.addItems(list(USES))
        self.speed_box = QComboBox()
        self.speed_box.addItems(["Any speed", "Fast", "Medium", "Slow"])
        self.direct_check = QCheckBox("One-click install only")
        self.direct_check.setChecked(True)
        self.commercial_check = QCheckBox("Hide non-commercial licences")
        for w in (self.search, self.scale_box, self.use_box, self.speed_box):
            top.addWidget(w, 3 if w is self.search else 0)
        top.addWidget(self.direct_check)
        top.addWidget(self.commercial_check)
        root.addLayout(top)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Model", "Scale", "Architecture", "Speed", "Size", "Licence", "Date", "Status"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        hh = self.table.horizontalHeader()
        hh.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)   # keep "new / updates first" until a header is clicked
        self.table.setSortingEnabled(True)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 8):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        split.addWidget(self.table)

        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(8, 0, 0, 0)
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(True)
        sl.addWidget(self.details, 1)
        self.install_btn = QPushButton("Install")
        self.install_btn.setObjectName("primary")
        self.page_btn = QPushButton("Open page on OpenModelDB ↗")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.hide()
        sl.addWidget(self.bar)
        sl.addWidget(self.install_btn)
        sl.addWidget(self.page_btn)
        side.setMinimumWidth(330)
        split.addWidget(side)
        split.setSizes([800, 360])
        root.addWidget(split, 1)

        bottom = QHBoxLayout()
        self.status = QLabel("Loading the catalogue...")
        self.status.setObjectName("muted")
        self.refresh_btn = QPushButton("Refresh catalogue")
        close = QPushButton("Close")
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.refresh_btn)
        bottom.addWidget(close)
        root.addLayout(bottom)

        self.search.textChanged.connect(self._filter)
        for w in (self.scale_box, self.use_box, self.speed_box):
            w.currentIndexChanged.connect(self._filter)
        for w in (self.direct_check, self.commercial_check):
            w.toggled.connect(self._filter)
        self.table.itemSelectionChanged.connect(self._show_details)
        self.install_btn.clicked.connect(self._install)
        self.page_btn.clicked.connect(lambda: self._current() and QDesktopServices.openUrl(QUrl(self._current().page_url)))
        self.refresh_btn.clicked.connect(lambda: self._load(force=True))
        close.clicked.connect(self.close)

    # ------------------------------------------------------------------ data
    def _load(self, force):
        self.refresh_btn.setEnabled(False)
        self.status.setText("Downloading the latest catalogue from OpenModelDB..." if force else "Loading the catalogue...")

        def work():
            from upscaler import catalog
            try:
                self.b.loaded.emit(catalog.load_catalog(force=force), None)
            except Exception as e:  # noqa: BLE001
                self.b.loaded.emit(None, str(e))
        threading.Thread(target=work, daemon=True).start()

    def _loaded(self, models, error):
        from upscaler import catalog
        self.refresh_btn.setEnabled(True)
        if error:
            self.status.setText(f"Couldn't load the catalogue: {error}")
            return
        self.models = [m for m in models if m.supported]
        self.installed = catalog.installed_ids()
        self.inst_map = catalog._installed_map()
        first_time = not self.seen
        self.new_ids = set() if first_time else {m.id for m in self.models} - self.seen
        self._filter()
        age = catalog.catalog_age() or 0
        when = "just now" if age < 90 else f"{age / 3600:.0f} h ago" if age >= 3600 else f"{age / 60:.0f} min ago"
        updates = sum(1 for m in self.models if catalog.status_of(m, self.installed, self.inst_map) == "update")
        extra = (f"  ·  {len(self.new_ids)} new since you last looked" if self.new_ids else "") + \
                (f"  ·  {updates} installed model(s) have updates" if updates else "")
        self.status.setText(f"{len(self.models)} models usable in MUSS  ·  catalogue updated {when}{extra}")

    def _matches(self, m):
        q = self.search.text().strip().lower()
        if q and q not in " ".join([m.id, m.name, m.author, m.description, " ".join(m.tags), m.arch_label]).lower():
            return False
        sc = self.scale_box.currentIndex()
        if sc and {1: m.scale == 2, 2: m.scale == 4, 3: m.scale == 1, 4: m.scale not in (1, 2, 4)}[sc] is False:
            return False
        use = USES[self.use_box.currentText()]
        if use and not use & set(m.tags):
            return False
        sp = self.speed_box.currentIndex()
        if sp and m.speed != ["", "fast", "medium", "slow"][sp]:
            return False
        if self.direct_check.isChecked() and not m.direct_url and m.id not in self.installed:
            return False
        if self.commercial_check.isChecked() and any(t in m.license for t in NON_COMMERCIAL):
            return False
        return True

    def _filter(self, *_):
        from upscaler import catalog
        sel = self._current()
        self.shown = [m for m in self.models if self._matches(m)]
        # new and updatable models first
        self.shown.sort(key=lambda m: (m.id not in self.new_ids,
                                       catalog.status_of(m, self.installed, self.inst_map) != "update"))
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.shown))
        for r, m in enumerate(self.shown):
            st = catalog.status_of(m, self.installed, self.inst_map)
            status = {"update": "Update available", "installed": "Installed"}.get(st) or \
                ("New" if m.id in self.new_ids else "" if m.direct_url else "Browser download")
            vals = [m.name, f"{m.scale}×", m.arch_label, m.speed, f"{m.size / 1e6:.1f} MB" if m.size else "?",
                    m.license, m.date, status]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setData(Qt.ItemDataRole.UserRole, r)
                if c == 7 and status:
                    it.setForeground(QColor({"Installed": "#3ecf8e", "Update available": "#f5b74f",
                                             "New": "#4f8cff"}.get(status, "#8b93a5")))
                if c == 4 and m.size:
                    it.setData(Qt.ItemDataRole.DisplayRole, f"{m.size / 1e6:6.1f} MB")
                self.table.setItem(r, c, it)
        self.table.setSortingEnabled(True)
        if sel and sel in self.shown:
            for r in range(self.table.rowCount()):
                if self._row_model(r) is sel:
                    self.table.selectRow(r)
                    break
        elif self.shown:
            self.table.selectRow(0)
        if self.shown:
            # the selected row index may be unchanged while its model changed: refresh explicitly
            self._show_details()
        else:
            self.details.setHtml("<p>No models match these filters.</p>")
            self.install_btn.setEnabled(False)

    def _row_model(self, row):
        it = self.table.item(row, 0)
        return self.shown[it.data(Qt.ItemDataRole.UserRole)] if it else None

    def _current(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self._row_model(rows[0].row()) if rows else None

    def _show_details(self):
        from upscaler import catalog
        m = self._current()
        if not m:
            return
        st = catalog.status_of(m, self.installed, self.inst_map)
        nc = any(t in m.license for t in NON_COMMERCIAL)
        speed_note = {"fast": "Fast enough for video.", "medium": "Moderate speed at 1080p.",
                      "slow": "Large/transformer model: very slow at 1080p."}[m.speed]
        desc = (m.description or "No description.").replace("\n", "<br>")
        self.details.setHtml(
            f"<h3 style='margin:0'>{m.name}</h3>"
            f"<p style='color:#8b93a5'>by {m.author or 'unknown'} · {m.date}</p>"
            f"<p><b>{m.scale}×</b> · {m.arch_label} · {m.size / 1e6:.1f} MB · {m.file_type}<br>{speed_note}"
            + (" Output is resized to 2× (4K) on the GPU." if m.scale not in (2,) else "") + "</p>"
            f"<p><b>Licence:</b> {m.license}" + (" <span style='color:#f5b74f'>(non-commercial use only)</span>" if nc else "")
            + f"</p><p><b>Tags:</b> {', '.join(m.tags)}</p><p>{desc}</p>"
            f"<p><a href='{m.page_url}'>View on OpenModelDB (sample images)</a></p>")
        if st == "installed":
            self.install_btn.setText("Installed")
            self.install_btn.setEnabled(False)
        elif st == "update":
            self.install_btn.setText("Update model")
            self.install_btn.setEnabled(bool(m.direct_url) and not self.busy)
        elif m.direct_url:
            self.install_btn.setText(f"Install  ({m.size / 1e6:.0f} MB)")
            self.install_btn.setEnabled(not self.busy)
        else:
            self.install_btn.setText("Download in browser, then drop the file on MUSS")
            self.install_btn.setEnabled(False)

    # ------------------------------------------------------------------ install
    def _install(self):
        from upscaler import catalog
        m = self._current()
        if not m or self.busy:
            return
        replace = catalog.status_of(m, self.installed, self.inst_map) == "update"
        self.busy, self.cancel_flag = True, False
        self.install_btn.setEnabled(False)
        self.install_btn.setText("Downloading...")
        self.bar.setValue(0)
        self.bar.show()

        def work():
            try:
                spec = catalog.install(m, progress=lambda d, t: self.b.progress.emit(d, t),
                                       cancel=lambda: self.cancel_flag, replace=replace)
                self.b.installed.emit(spec, None)
            except Exception as e:  # noqa: BLE001
                self.b.installed.emit(None, str(e))
        threading.Thread(target=work, daemon=True).start()

    def _progress(self, done, total):
        if total:
            self.bar.setValue(int(done * 1000 / total))
            self.install_btn.setText(f"Downloading...  {done / 1e6:.0f} / {total / 1e6:.0f} MB")

    def _installed(self, spec, error):
        from upscaler import catalog
        self.busy = False
        self.bar.hide()
        if error:
            if error != "cancelled":
                QMessageBox.warning(self, "Model library", f"The model was not installed:\n\n{error}")
        else:
            self.status.setText(f"Installed {spec.label} - selected in MUSS.")
            self.on_installed(spec)
        self.installed = catalog.installed_ids()
        self.inst_map = catalog._installed_map()
        self._filter()
        self._show_details()

    def closeEvent(self, e):
        if self.busy:
            self.cancel_flag = True
        if self.models:   # everything in the catalogue now counts as "seen"
            self.settings.setValue("catalog_seen", json.dumps(sorted({m.id for m in self.models} | self.seen)))
        e.accept()
