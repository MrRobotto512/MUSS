"""MUSS (Movie UpScaling Software) - desktop app (Qt). Upscales video 2x (1080p -> 4K) on one or more NVIDIA GPUs."""
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, QPointF, QRectF, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QFont, QImage, QKeySequence, QPainter, QPainterPath,
                           QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDockWidget, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter, QToolButton, QVBoxLayout,
    QTimeEdit, QWidget,
)
from PySide6.QtCore import QTime

from timeline import RangeTimeline, fmt_tc

APP_NAME = "MUSS"
APP_TITLE = "MUSS - Movie UpScaling Software"
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".ts", ".m2ts", ".wmv", ".mts")
AUDIO_EXTS = (".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".ac3", ".eac3", ".dts",
              ".mka", ".aif", ".aiff", ".alac", ".ape", ".wv", ".mp2")
UPSCALE_MODES = ["Video only", "Video + audio", "Audio only (keep picture)"]
AUDIO_QUALITY = [("Standard (25 steps)", 25), ("High (50 steps, 2× slower)", 50)]

# --------------------------------------------------------------------------- theme
C_BG, C_PANEL, C_PANEL2, C_BORDER = "#121419", "#1a1d24", "#22262f", "#2c313c"
C_TEXT, C_MUTED, C_ACCENT, C_ACCENT_H = "#e6e8ee", "#8b93a5", "#4f8cff", "#6aa0ff"
C_OK, C_WARN, C_ERR = "#3ecf8e", "#f5b74f", "#ff6b6b"

STYLE = f"""
* {{ font-family: 'Segoe UI'; font-size: 10pt; color: {C_TEXT}; }}
QMainWindow, QWidget#central {{ background: {C_BG}; }}
QFrame#panel {{ background: {C_PANEL}; border: 1px solid {C_BORDER}; border-radius: 10px; }}
QFrame#topbar {{ background: {C_PANEL}; border-bottom: 1px solid {C_BORDER}; }}
QLabel#title {{ font-size: 13pt; font-weight: 600; }}
QLabel#section {{ color: {C_MUTED}; font-size: 8.5pt; font-weight: 600; letter-spacing: 1px; padding-top: 6px; }}
QLabel#muted {{ color: {C_MUTED}; font-size: 9pt; }}
QLabel#badge {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 9px; padding: 2px 9px; font-size: 8.5pt; }}
QLabel#badgeOn {{ background: #1d3a2e; border: 1px solid #2c6b4f; color: {C_OK}; border-radius: 9px; padding: 2px 9px; font-size: 8.5pt; }}
QPushButton {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 7px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {C_ACCENT}; }}
QPushButton:disabled {{ color: #555c6b; border-color: #262a33; }}
QPushButton#primary {{ background: {C_ACCENT}; border: none; font-weight: 600; padding: 10px; font-size: 11pt; }}
QPushButton#primary:hover {{ background: {C_ACCENT_H}; }}
QPushButton#primary:disabled {{ background: #2a3a5c; color: #8090b0; }}
QPushButton#danger {{ background: #3a1f24; border: 1px solid #6b2c35; color: {C_ERR}; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 4px 9px; color: {C_MUTED}; }}
QToolButton:hover {{ color: {C_TEXT}; background: {C_PANEL2}; }}
QToolButton:checked {{ color: {C_TEXT}; background: {C_PANEL2}; border-color: {C_BORDER}; }}
QComboBox, QLineEdit {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 7px; padding: 5px 8px; }}
QComboBox:hover, QLineEdit:hover {{ border-color: #3a4150; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; selection-background-color: {C_ACCENT}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #485063; background: {C_PANEL2}; }}
QCheckBox::indicator:checked {{ background: {C_ACCENT}; border-color: {C_ACCENT}; }}
QCheckBox:disabled {{ color: #555c6b; }}
QSlider::groove:horizontal {{ height: 4px; background: {C_BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {C_ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #fff; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QProgressBar {{ background: {C_PANEL2}; border: none; border-radius: 3px; height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {C_ACCENT}; border-radius: 3px; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border-radius: 8px; margin: 2px 0; }}
QListWidget::item:selected {{ background: {C_PANEL2}; border: 1px solid {C_ACCENT}; }}
QListWidget::item:hover:!selected {{ background: #1f232b; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {C_BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QPlainTextEdit {{ background: {C_BG}; border: none; font-family: Consolas; font-size: 9pt; color: #b8bfcc; }}
QDockWidget {{ color: {C_MUTED}; }}
QDockWidget::title {{ background: {C_PANEL}; padding: 4px 8px; }}
QStatusBar {{ background: {C_PANEL}; border-top: 1px solid {C_BORDER}; }}
QSplitter::handle {{ background: {C_BG}; }}
QToolTip {{ background: {C_PANEL2}; color: {C_TEXT}; border: 1px solid {C_BORDER}; padding: 4px; }}
"""


def fmt_time(s):
    s = int(max(s or 0, 0))
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def rgb_to_image(buf: bytes, w: int, h: int) -> QImage:
    return QImage(buf, w, h, w * 3, QImage.Format.Format_RGB888).copy()


def make_icon():
    """The MUSS icon, drawn in code (blue rounded square with a white 'M' and a 4K-style corner mark)."""
    from PySide6.QtGui import QIcon
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(C_ACCENT))
        r = size * 0.2
        p.drawRoundedRect(QRectF(0, 0, size, size), r, r)
        f = QFont("Segoe UI")
        f.setBold(True)
        f.setPixelSize(int(size * 0.62))
        p.setFont(f)
        p.setPen(QColor("#ffffff"))
        p.drawText(QRectF(0, -size * 0.04, size, size), Qt.AlignmentFlag.AlignCenter, "M")
        p.setBrush(QColor("#ffffff"))
        p.setPen(Qt.PenStyle.NoPen)
        d = size * 0.16
        p.drawEllipse(QRectF(size - d * 1.5, size - d * 1.5, d, d))
        p.end()
        icon.addPixmap(pm)
    return icon


RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def startup_command() -> str:
    """Command Windows runs at login: MUSS.exe from an installer install, else this app.py with pythonw."""
    home = os.environ.get("MUSS_HOME")
    if home and os.path.isfile(os.path.join(home, "MUSS.exe")):
        return f'"{os.path.join(home, "MUSS.exe")}" --tray'
    exe = sys.executable
    if exe.lower().endswith("python.exe") and os.path.isfile(exe[:-10] + "pythonw.exe"):
        exe = exe[:-10] + "pythonw.exe"
    return f'"{exe}" "{os.path.abspath(__file__)}" --tray'


def startup_enabled() -> bool:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, APP_NAME)
            return True
    except OSError:
        return False


def set_startup(on: bool):
    """Add/remove MUSS in the current user's startup programs (HKCU Run key; no admin rights needed)."""
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        if on:
            winreg.SetValueEx(k, APP_NAME, 0, winreg.REG_SZ, startup_command())
        else:
            try:
                winreg.DeleteValue(k, APP_NAME)
            except FileNotFoundError:
                pass


def wrap_image(buf: bytes, w: int, h: int) -> QImage:
    """Zero-copy view of RGB bytes; only valid while buf is alive (convert to QPixmap right away)."""
    return QImage(buf, w, h, w * 3, QImage.Format.Format_RGB888)


@dataclass
class PreviewClip:
    """A rendered preview range held in memory: original + AI frame for every frame in the range."""
    path: str
    model_key: str
    start: float
    fps: float
    w: int
    h: int
    total: int
    src: list = field(default_factory=list)
    ai: list = field(default_factory=list)
    complete: bool = False
    job: object = None

    def index_at(self, t):
        k = int(round((t - self.start) * self.fps))
        return k if 0 <= k < len(self.ai) else None

    def time_of(self, k):
        return self.start + k / self.fps

    @property
    def end(self):
        return self.start + self.total / self.fps


# --------------------------------------------------------------------------- data
@dataclass
class QueueItem:
    path: str
    info: object = None
    status: str = "Loading"          # Loading, Ready, Queued, Processing, Done, Failed, Cancelled
    progress: float = 0.0
    detail: str = ""
    output_path: str = ""
    thumb: QImage | None = None
    settings: dict = field(default_factory=dict)
    list_item: QListWidgetItem | None = None
    widget: "QueueRow | None" = None
    playhead: float = -1.0           # seconds; -1 = not set yet
    r_in: float = -1.0               # preview range (seconds)
    r_out: float = -1.0
    thumbs: list = field(default_factory=list)
    kind: str = "video"              # "video" or "audio" (audio-only files)
    ainfo: object = None             # AudioInfo of the first audio track
    cutoff: float | None = None      # measured treble cutoff (Hz), None = not measured yet
    analyzing: bool = False
    restore: dict | None = None      # saved queue state, applied once the file has been read


class Bridge(QObject):
    """Signals emitted from worker threads, delivered on the UI thread."""
    engine_ready = Signal(object)
    engine_failed = Signal(str)
    item_loaded = Signal(object)
    item_changed = Signal(object)
    progress = Signal(object, object)
    log = Signal(str)
    frame_ready = Signal(object)
    ai_ready = Signal(object)
    export_idle = Signal()
    thumbs_ready = Signal(object, object)
    clip_progress = Signal(object)
    clip_done = Signal(object, object)
    warm_done = Signal(object)
    models_changed = Signal(object, object)    # (key to select or None, error or None)
    nodes_status = Signal(object)              # {node key: info dict or error string}
    update_checked = Signal(object, object, bool)   # (release or None, error or None, manual)
    update_done = Signal(object, object)            # (new version or None, error or None)
    catalog_checked = Signal(int, int)              # (new models, installed models with updates)
    audio_analyzed = Signal(object)                 # QueueItem whose audio cutoff was measured
    audio_model_progress = Signal(int, int, object)  # (done, total, error or None / "done")


# --------------------------------------------------------------------------- queue row
class QueueRow(QWidget):
    def __init__(self, item: QueueItem):
        super().__init__()
        self.item = item
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)
        self.thumb = QLabel()
        self.thumb.setFixedSize(96, 54)
        self.thumb.setStyleSheet(f"background: #0d0f13; border-radius: 5px;")
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.thumb)
        col = QVBoxLayout()
        col.setSpacing(3)
        self.name = QLabel(os.path.basename(item.path))
        self.name.setStyleSheet("font-weight: 600;")
        self.name.setMinimumWidth(10)
        self.meta = QLabel("Reading...")
        self.meta.setObjectName("muted")
        self.status = QLabel("")
        self.status.setObjectName("muted")
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(5)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        for wdg in (self.name, self.meta, self.status, self.bar):
            col.addWidget(wdg)
        lay.addLayout(col, 1)
        self.refresh()

    def refresh(self):
        it = self.item
        fm = self.name.fontMetrics()
        self.name.setText(fm.elidedText(os.path.basename(it.path), Qt.TextElideMode.ElideMiddle, 170))
        self.name.setToolTip(it.path)
        if it.info:
            i = it.info
            self.meta.setText(f"{i.width}×{i.height}  ·  {fmt_time(i.duration)}")
            self.meta.setToolTip(f"{float(i.fps):.3f} fps  ·  {i.frames} frames  ·  audio: {i.audio_codec or 'none'}")
        elif it.kind == "audio" and it.ainfo:
            a = it.ainfo
            ch = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}.get(a.channels, f"{a.channels} ch")
            self.meta.setText(f"Audio  ·  {a.codec.upper()} {ch}  ·  {fmt_time(a.duration)}")
            if self.thumb.pixmap() is None or self.thumb.pixmap().isNull():
                self.thumb.setText("♪")
                self.thumb.setStyleSheet(f"background: #0d0f13; border-radius: 5px; color: {C_ACCENT}; font-size: 22pt;")
        if it.thumb is not None:
            self.thumb.setPixmap(QPixmap.fromImage(it.thumb).scaled(
                96, 54, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        colors = {"Done": C_OK, "Failed": C_ERR, "Processing": C_ACCENT, "Queued": C_WARN, "Stopped": C_WARN,
                  "Skipped": C_MUTED,
                  "Scheduled": "#b18cff"}
        color = colors.get(it.status, C_MUTED)
        text = it.status if not it.detail else f"{it.status} · {it.detail}"
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {color}; font-size: 9pt;")
        self.status.setToolTip(it.output_path if it.status == "Done" else it.detail)
        self.bar.setVisible(it.status == "Processing")
        self.bar.setValue(int(it.progress * 1000))


# --------------------------------------------------------------------------- viewer
class CompareView(QWidget):
    """Before/after viewer: split slider, side-by-side, zoom (wheel) and pan (drag)."""

    MODES = ("Split", "Side by side", "Original", "Enhanced")

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setMinimumSize(400, 250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # QImages are drawn directly (no per-frame QPixmap conversion) so 4K playback stays smooth.
        self.before: QImage | None = None
        self.after: QImage | None = None
        self._keep = None               # keeps the raw buffers alive for zero-copy QImages
        self.fast = False               # True during playback: cheaper scaling
        self.img_w = self.img_h = 0     # output (2x) image size
        self.mode = "Split"
        self.split = 0.5
        self.zoom = None                # None = fit
        self.pan = QPointF(0, 0)
        self._drag = None
        self.placeholder = "Drop videos here or click  + Add videos"
        self.busy_text = ""
        self.on_zoom_changed = lambda: None

    # ---- content
    def set_frames(self, before: QImage | None, after: QImage | None, out_w=0, out_h=0, keep=None):
        if before is not None:
            self.before = before
            self.img_w, self.img_h = out_w, out_h
        elif after is None:
            self.before = None
        self.after = after
        self._keep = keep
        self.update()
        self.on_zoom_changed()

    def set_after(self, after: QImage | None):
        self.after = after
        self.update()

    def set_fast(self, fast: bool):
        if self.fast != fast:
            self.fast = fast
            self.update()

    def set_mode(self, mode):
        self.mode = mode
        self.update()

    def set_zoom(self, z):
        self.zoom = z
        if z is None:
            self.pan = QPointF(0, 0)
        self.update()
        self.on_zoom_changed()

    # ---- geometry
    def _areas(self):
        r = QRectF(self.rect())
        if self.mode == "Side by side":
            half = r.width() / 2
            return [QRectF(r.left(), r.top(), half - 1, r.height()), QRectF(r.left() + half + 1, r.top(), half - 1, r.height())]
        return [r]

    def _fit_zoom(self, area: QRectF):
        if not self.img_w:
            return 1.0
        return min((area.width() - 24) / self.img_w, (area.height() - 24) / self.img_h)

    def current_zoom(self):
        return self.zoom if self.zoom is not None else self._fit_zoom(self._areas()[0])

    def _image_rect(self, area: QRectF) -> QRectF:
        z = self.current_zoom()
        w, h = self.img_w * z, self.img_h * z
        c = area.center() + self.pan
        return QRectF(c.x() - w / 2, c.y() - h / 2, w, h)

    # ---- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0b0d11"))
        if self.before is None:
            p.setPen(QColor(C_MUTED))
            f = p.font()
            f.setPointSize(12)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.busy_text or self.placeholder)
            return
        z = self.current_zoom()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        areas = self._areas()
        if self.mode == "Side by side":
            for area, pix, label in ((areas[0], self.before, "Original"), (areas[1], self.after, "AI Enhanced")):
                p.save()
                p.setClipRect(area)
                self._draw(p, pix or self.before, self._image_rect(area), z, enhanced=pix is not None and label != "Original")
                p.restore()
                self._label(p, area, label, left=True, missing=(label != "Original" and self.after is None))
            p.fillRect(QRectF(areas[0].right(), 0, 2, self.height()), QColor(C_BORDER))
        else:
            area = areas[0]
            rect = self._image_rect(area)
            if self.mode == "Original" or (self.mode == "Enhanced" and self.after is None):
                self._draw(p, self.before, rect, z)
                self._label(p, area, "Original", left=True)
                if self.mode == "Enhanced":
                    self._label(p, area, "AI Enhanced", left=False, missing=True)
            elif self.mode == "Enhanced":
                self._draw(p, self.after, rect, z, enhanced=True)
                self._label(p, area, "AI Enhanced", left=False)
            else:  # split
                sx = area.left() + area.width() * self.split
                p.save()
                p.setClipRect(QRectF(area.left(), area.top(), sx - area.left(), area.height()))
                self._draw(p, self.before, rect, z)
                p.restore()
                p.save()
                p.setClipRect(QRectF(sx, area.top(), area.right() - sx, area.height()))
                self._draw(p, self.after or self.before, rect, z, enhanced=self.after is not None)
                p.restore()
                p.setPen(QPen(QColor("#ffffff"), 2))
                p.drawLine(QPointF(sx, area.top()), QPointF(sx, area.bottom()))
                # handle
                p.setBrush(QColor("#ffffff"))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QPointF(sx, area.center().y()), 14, 14)
                p.setPen(QPen(QColor("#20242c"), 2))
                cy = area.center().y()
                for d in (-1, 1):
                    path = QPainterPath()
                    path.moveTo(sx + d * 4, cy - 5)
                    path.lineTo(sx + d * 9, cy)
                    path.lineTo(sx + d * 4, cy + 5)
                    p.drawPath(path)
                self._label(p, area, "Original", left=True)
                self._label(p, area, "AI Enhanced", left=False, missing=self.after is None)
        if self.busy_text:
            self._pill(p, QPointF(self.width() / 2, self.height() - 34), self.busy_text)

    def _draw(self, p, pix, rect, z, enhanced=False):
        # The original frame is half the output size: drawing it into the same rect upscales it
        # with plain smoothing, which is exactly the "before" comparison.
        smooth = not self.fast and not (enhanced and z >= 1.0)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
        # Only draw the visible part of the image (big win when zoomed in on 4K frames).
        vis = rect.intersected(QRectF(p.clipBoundingRect()) if p.hasClipping() else QRectF(self.rect()))
        if vis.isEmpty():
            return
        sx, sy = pix.width() / rect.width(), pix.height() / rect.height()
        src = QRectF((vis.left() - rect.left()) * sx, (vis.top() - rect.top()) * sy, vis.width() * sx, vis.height() * sy)
        p.drawImage(vis, pix, src)

    def _label(self, p, area, text, left, missing=False):
        if missing:
            text += "  ·  not rendered"
        f = p.font()
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        tw = p.fontMetrics().horizontalAdvance(text) + 20
        x = area.left() + 12 if left else area.right() - 12 - tw
        r = QRectF(x, area.top() + 12, tw, 26)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 160))
        p.drawRoundedRect(r, 13, 13)
        p.setPen(QColor(C_MUTED if missing else "#ffffff"))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

    def _pill(self, p, center, text):
        f = p.font()
        f.setPointSize(10)
        f.setBold(False)
        p.setFont(f)
        tw = p.fontMetrics().horizontalAdvance(text) + 28
        r = QRectF(center.x() - tw / 2, center.y() - 15, tw, 30)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(20, 24, 32, 220))
        p.drawRoundedRect(r, 15, 15)
        p.setPen(QColor("#ffffff"))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

    # ---- interaction
    def _split_x(self):
        a = self._areas()[0]
        return a.left() + a.width() * self.split

    def mousePressEvent(self, e):
        if self.before is None or e.button() != Qt.MouseButton.LeftButton:
            return
        if self.mode == "Split" and abs(e.position().x() - self._split_x()) < 18:
            self._drag = ("split", None)
        else:
            self._drag = ("pan", (e.position(), QPointF(self.pan)))
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        x = e.position().x()
        if self._drag and self._drag[0] == "split":
            a = self._areas()[0]
            self.split = min(max((x - a.left()) / max(a.width(), 1), 0.0), 1.0)
            self.update()
        elif self._drag and self._drag[0] == "pan":
            start, pan0 = self._drag[1]
            self.pan = pan0 + (e.position() - start)
            if self.zoom is None:
                self.zoom = self.current_zoom()
                self.on_zoom_changed()
            self.update()
        elif self.mode == "Split" and self.before is not None and abs(x - self._split_x()) < 18:
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.before is not None else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, _):
        self._drag = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.on_zoom_changed()

    def mouseDoubleClickEvent(self, _):
        self.set_zoom(None if self.zoom is not None else 1.0)

    def wheelEvent(self, e):
        if self.before is None:
            return
        old = self.current_zoom()
        new = min(max(old * (1.2 ** (e.angleDelta().y() / 120)), 0.05), 8.0)
        area = self._areas()[0]
        if self.mode == "Side by side" and e.position().x() > area.right():
            area = self._areas()[1]
        rect = self._image_rect(area)
        cur = e.position()
        u = (cur - rect.topLeft()) / old
        new_tl = cur - u * new
        self.pan = new_tl - area.center() + QPointF(self.img_w * new / 2, self.img_h * new / 2)
        self.zoom = new
        self.update()
        self.on_zoom_changed()


# --------------------------------------------------------------------------- main window
PREVIEW_LENGTHS = [("1 s", 1), ("2 s", 2), ("3 s", 3), ("5 s", 5), ("10 s", 10), ("Custom", 0)]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 940)
        self.setAcceptDrops(True)
        self.settings = QSettings("AIUpscaler", "AIUpscaler")
        self.bridge = Bridge()
        self.engine = self.models = self.trt = None
        self.previewer = None
        self.items: list[QueueItem] = []
        self.current: QueueItem | None = None
        self.gpu_checks: dict[int, QCheckBox] = {}
        self.export_q: queue.Queue = queue.Queue()
        self.export_thread = None
        self.exporting = False
        self.current_job = None
        self.scheduled: list = []        # videos waiting for the schedule's time window
        self._sched_pausing = False      # the window ended: current job is being paused
        self._sched_override = False     # "Start scheduled now" pressed: ignore the window until idle
        self.preview_req = 0
        self.frame_cache = {}          # (path, frame index) -> (QImage, rgb bytes)
        self.preview_buf = None
        self.clip: PreviewClip | None = None
        self.playing = False
        self.warming = False
        self._shown_clip_frame = None

        self._build_ui()
        self._connect()
        self.setWindowIcon(make_icon())
        self._really_quit = False
        self._tray_hint_shown = False
        self._setup_tray()
        self.save_timer = QTimer(self)               # debounced queue saving
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(800)
        self.save_timer.timeout.connect(self._save_queue)
        self._set_enabled(False)
        self.log("Starting AI engine (PyTorch)...")
        threading.Thread(target=self._load_engine, daemon=True).start()

    # ================================================================ UI construction
    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- top bar
        top = QFrame()
        top.setObjectName("topbar")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(16, 8, 16, 8)
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        tl.addWidget(title)
        tl.addSpacing(18)
        self.add_btn = QPushButton("+  Add files")
        tl.addWidget(self.add_btn)
        tl.addStretch(1)
        self.gpu_badge = QLabel("Detecting GPUs...")
        self.gpu_badge.setObjectName("badge")
        self.trt_badge = QLabel("TensorRT")
        self.trt_badge.setObjectName("badge")
        self.log_btn = QToolButton()
        self.log_btn.setText("Log")
        self.log_btn.setCheckable(True)
        from upscaler import __version__ as muss_version
        self.update_btn = QPushButton("Update available")
        self.update_btn.setObjectName("primary")
        self.update_btn.setStyleSheet("padding: 4px 12px; font-size: 9.5pt;")
        self.update_btn.hide()
        self.version_btn = QToolButton()
        self.version_btn.setText(f"v{muss_version}")
        self.version_btn.setToolTip("Check for updates")
        tl.addWidget(self.update_btn)
        tl.addWidget(self.version_btn)
        self.node_mode_btn = QToolButton()
        self.node_mode_btn.setText("Render node: off")
        self.node_mode_btn.setCheckable(True)
        self.node_mode_btn.setToolTip("Let another PC running MUSS use this PC's GPUs")
        for wdg in (self.node_mode_btn, self.gpu_badge, self.trt_badge, self.log_btn):
            tl.addWidget(wdg)
        root.addWidget(top)

        body = QSplitter(Qt.Orientation.Horizontal)
        body.setHandleWidth(8)
        body.setContentsMargins(8, 8, 8, 8)
        root.addWidget(body, 1)

        # ---- queue panel
        qp = QFrame()
        qp.setObjectName("panel")
        ql = QVBoxLayout(qp)
        ql.setContentsMargins(10, 10, 10, 10)
        hdr = QHBoxLayout()
        lab = QLabel("QUEUE")
        lab.setObjectName("section")
        hdr.addWidget(lab)
        hdr.addStretch(1)
        self.remove_btn = QToolButton()
        self.remove_btn.setText("Remove")
        self.clear_btn = QToolButton()
        self.clear_btn.setText("Clear done")
        hdr.addWidget(self.remove_btn)
        hdr.addWidget(self.clear_btn)
        ql.addLayout(hdr)
        self.queue_list = QListWidget()
        self.queue_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.queue_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        ql.addWidget(self.queue_list, 1)
        self.queue_hint = QLabel("Drag video files anywhere\ninto this window.")
        self.queue_hint.setObjectName("muted")
        self.queue_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ql.addWidget(self.queue_hint)
        qp.setMinimumWidth(250)
        body.addWidget(qp)

        # ---- viewer panel
        vp = QFrame()
        vp.setObjectName("panel")
        vl = QVBoxLayout(vp)
        vl.setContentsMargins(10, 10, 10, 8)
        vl.setSpacing(6)
        vbar = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        for m in CompareView.MODES:
            b = QToolButton()
            b.setText(m)
            b.setCheckable(True)
            b.setChecked(m == "Split")
            self.mode_group.addButton(b)
            vbar.addWidget(b)
        vbar.addStretch(1)
        self.zoom_label = QLabel("")
        self.zoom_label.setObjectName("muted")
        vbar.addWidget(self.zoom_label)
        self.zoom_btns = {}
        for text, z in (("Fit", None), ("100%", 1.0), ("200%", 2.0)):
            b = QToolButton()
            b.setText(text)
            self.zoom_btns[text] = (b, z)
            vbar.addWidget(b)
        vl.addLayout(vbar)
        self.view = CompareView()
        vl.addWidget(self.view, 1)

        self.timeline = RangeTimeline()
        vl.addWidget(self.timeline)

        tr = QHBoxLayout()
        tr.setSpacing(4)

        def tbtn(text, tip):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setStyleSheet("font-size: 11pt; padding: 3px 8px;")
            tr.addWidget(b)
            return b

        self.to_in_btn = tbtn("⇤", "Go to preview start (Home)")
        self.step_back_btn = tbtn("|◀", "Previous frame (Left arrow)")
        self.play_btn = tbtn("▶", "Play / pause the preview (Space)")
        self.step_fwd_btn = tbtn("▶|", "Next frame (Right arrow)")
        self.to_out_btn = tbtn("⇥", "Go to preview end (End)")
        self.tc_label = QLabel("00:00:00:00")
        self.tc_label.setStyleSheet("font-family: Consolas; font-size: 10.5pt; padding: 0 6px;")
        self.tc_label.setMinimumWidth(self.tc_label.fontMetrics().horizontalAdvance("00:00:00:00") + 24)
        tr.addWidget(self.tc_label)
        self.dur_label = QLabel("/ 00:00:00:00")
        self.dur_label.setObjectName("muted")
        tr.addWidget(self.dur_label)
        tr.addStretch(1)
        self.set_in_btn = QPushButton("[  In")
        self.set_in_btn.setToolTip("Start the preview range at the playhead (I)")
        self.set_out_btn = QPushButton("Out  ]")
        self.set_out_btn.setToolTip("End the preview range at the playhead (O)")
        tr.addWidget(self.set_in_btn)
        tr.addWidget(self.set_out_btn)
        self.len_box = QComboBox()
        for label, _ in PREVIEW_LENGTHS:
            self.len_box.addItem(label)
        self.len_box.setToolTip("Preview length")
        self.len_box.setFixedWidth(84)
        tr.addWidget(self.len_box)
        self.range_label = QLabel("")
        self.range_label.setObjectName("muted")
        self.range_label.setMinimumWidth(110)
        tr.addWidget(self.range_label)
        self.frame_btn = QPushButton("Frame")
        self.frame_btn.setToolTip("Render the AI result for just this frame (F)")
        self.preview_btn = QPushButton("▶  Preview")
        self.preview_btn.setObjectName("primary")
        self.preview_btn.setStyleSheet("padding: 6px 16px; font-size: 10pt;")
        self.preview_btn.setToolTip("Render the AI result for the whole preview range, then play it (P)")
        tr.addWidget(self.frame_btn)
        tr.addWidget(self.preview_btn)
        vl.addLayout(tr)
        body.addWidget(vp)

        # ---- settings panel
        sp = QFrame()
        sp.setObjectName("panel")
        spl = QVBoxLayout(sp)
        spl.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        s = QVBoxLayout(inner)
        s.setContentsMargins(14, 10, 14, 14)
        s.setSpacing(7)
        scroll.setWidget(inner)
        spl.addWidget(scroll, 1)

        def section(text):
            lbl = QLabel(text.upper())
            lbl.setObjectName("section")
            s.addSpacing(6)
            s.addWidget(lbl)

        def muted(text=""):
            lbl = QLabel(text)
            lbl.setObjectName("muted")
            lbl.setWordWrap(True)
            s.addWidget(lbl)
            return lbl

        section("Output")
        self.out_res = QLabel("Select a video")
        self.out_res.setStyleSheet("font-size: 12pt; font-weight: 600;")
        s.addWidget(self.out_res)
        self.out_res_sub = muted("Upscales 2× - 1920×1080 becomes 3840×2160 (4K UHD).")

        section("AI model")
        self.model_box = QComboBox()
        self.model_box.setMaxVisibleItems(20)
        s.addWidget(self.model_box)
        self.model_desc = muted()
        self.library_btn = QPushButton("Browse model library")
        self.library_btn.setToolTip("Browse ~600 free open-source models from OpenModelDB and install them in one click")
        s.addWidget(self.library_btn)
        mrow = QHBoxLayout()
        mrow.setSpacing(4)
        self.add_model_btn = QPushButton("+ Add file")
        self.add_model_btn.setToolTip("Add an upscaling model file (.pth / .safetensors). You can also drop "
                                      "model files onto the window.")
        self.remove_model_btn = QToolButton()
        self.remove_model_btn.setText("Remove")
        self.remove_model_btn.setToolTip("Remove the selected custom model")
        self.models_folder_btn = QToolButton()
        self.models_folder_btn.setText("Folder")
        self.models_folder_btn.setToolTip("Open the custom models folder")
        self.find_models_btn = QToolButton()
        self.find_models_btn.setText("Find ↗")
        self.find_models_btn.setToolTip("Browse free community models on OpenModelDB (opens your browser)")
        self.find_models_btn.hide()                      # superseded by the in-app library
        mrow.addWidget(self.add_model_btn)
        mrow.addWidget(self.remove_model_btn)
        mrow.addWidget(self.models_folder_btn)
        mrow.addStretch(1)
        s.addLayout(mrow)

        section("Audio")
        arow = QHBoxLayout()
        alab = QLabel("Upscale")
        alab.setFixedWidth(60)
        self.upscale_mode_box = QComboBox()
        self.upscale_mode_box.addItems(UPSCALE_MODES)
        self.upscale_mode_box.setToolTip("For videos: upscale the picture, the picture and the audio, or only the "
                                         "audio (the picture is copied unchanged). Audio files are always audio.")
        arow.addWidget(alab)
        arow.addWidget(self.upscale_mode_box, 1)
        s.addLayout(arow)
        qrow2 = QHBoxLayout()
        qlab2 = QLabel("Quality")
        qlab2.setFixedWidth(60)
        self.audio_quality_box = QComboBox()
        self.audio_quality_box.addItems([q for q, _ in AUDIO_QUALITY])
        qrow2.addWidget(qlab2)
        qrow2.addWidget(self.audio_quality_box, 1)
        s.addLayout(qrow2)
        frow2 = QHBoxLayout()
        flab = QLabel("Audio files")
        flab.setFixedWidth(60)
        self.audio_format_box = QComboBox()
        self.audio_format_box.setToolTip("Format for restored audio files. In videos the format follows the "
                                         "container: FLAC in MKV, ALAC in MOV, 320 kbps AAC in MP4.")
        frow2.addWidget(flab)
        frow2.addWidget(self.audio_format_box, 1)
        s.addLayout(frow2)
        self.audio_hint = muted("")
        self.audio_model_btn = QPushButton("Download AudioSR model (6.2 GB)")
        self.audio_model_btn.hide()
        s.addWidget(self.audio_model_btn)

        section("Acceleration")
        self.gpu_box = QVBoxLayout()
        self.gpu_box.setSpacing(5)
        s.addLayout(self.gpu_box)
        self.gpu_hint = muted("Frames are shared between GPUs; faster cards automatically take more.")
        self.trt_check = QCheckBox("Use NVIDIA TensorRT")
        s.addWidget(self.trt_check)
        self.trt_hint = muted()
        self.trt_retry_btn = QToolButton()
        self.trt_retry_btn.setText("Retry TensorRT for this model")
        self.trt_retry_btn.hide()
        s.addWidget(self.trt_retry_btn)

        section("Render nodes (other PCs)")
        self.nodes_box = QVBoxLayout()
        self.nodes_box.setSpacing(4)
        s.addLayout(self.nodes_box)
        nrow = QHBoxLayout()
        self.add_node_btn = QPushButton("+ Add node...")
        self.add_node_btn.setToolTip("Use the GPUs of another PC on your network. On that PC, open this app and "
                                     "switch on 'Render node' (top bar), or run 'Render Node.bat'.")
        nrow.addWidget(self.add_node_btn)
        nrow.addStretch(1)
        s.addLayout(nrow)
        self.nodes_hint = muted("Optional. Other PCs render segments of each export alongside this one.")

        section("Export")
        row = QHBoxLayout()
        self.encoder_box = QComboBox()
        self.container_box = QComboBox()
        self.container_box.addItems(["mp4", "mkv", "mov"])
        self.container_box.setFixedWidth(70)
        row.addWidget(self.encoder_box, 1)
        row.addWidget(self.container_box)
        s.addLayout(row)
        qrow = QHBoxLayout()
        qlab = QLabel("Quality")
        self.cq_slider = QSlider(Qt.Orientation.Horizontal)
        self.cq_slider.setRange(10, 35)
        self.cq_slider.setInvertedAppearance(True)
        self.cq_value = QLabel()
        self.cq_value.setFixedWidth(110)
        self.cq_value.setObjectName("muted")
        qrow.addWidget(qlab)
        qrow.addWidget(self.cq_slider, 1)
        qrow.addWidget(self.cq_value)
        s.addLayout(qrow)
        frow = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Same folder as the source video")
        self.folder_btn = QPushButton("...")
        self.folder_btn.setFixedWidth(36)
        frow.addWidget(self.folder_edit, 1)
        frow.addWidget(self.folder_btn)
        s.addLayout(frow)
        self.range_check = QCheckBox("Export only the preview range")
        self.range_check.setToolTip("Each video is trimmed to its own in/out range on the timeline.")
        s.addWidget(self.range_check)
        self.test_check = QCheckBox("Quick test - first 10 seconds only")
        s.addWidget(self.test_check)

        section("Schedule")
        self.sched_check = QCheckBox("Only run the queue between")
        self.sched_check.setToolTip("Exports you start outside this time window wait for it, e.g. overnight.\n"
                                    "When the window ends, the current video pauses (finished segments are kept)\n"
                                    "and continues automatically in the next window.")
        s.addWidget(self.sched_check)
        trow = QHBoxLayout()
        self.sched_start = QTimeEdit()
        self.sched_end = QTimeEdit()
        for te in (self.sched_start, self.sched_end):
            te.setDisplayFormat("HH:mm")
            te.setFixedWidth(80)
        trow.addWidget(self.sched_start)
        and_lbl = QLabel("and")
        and_lbl.setObjectName("muted")
        trow.addWidget(and_lbl)
        trow.addWidget(self.sched_end)
        trow.addStretch(1)
        s.addLayout(trow)
        self.sched_hint = muted("")

        section("App")
        self.startup_check = QCheckBox("Start MUSS when Windows starts (in the tray)")
        self.startup_check.setToolTip("MUSS starts hidden in the system tray when you log in, so a scheduled queue\n"
                                      "runs without opening anything. Untick to remove it from startup.")
        s.addWidget(self.startup_check)
        self.tray_check = QCheckBox("Keep running in the tray when the window is closed")
        self.tray_check.setToolTip("Closing the window hides MUSS to the system tray (next to the clock) instead of\n"
                                   "quitting, so exports and the schedule keep going. Quit from the tray menu.")
        s.addWidget(self.tray_check)
        urow = QHBoxLayout()
        self.auto_update_check = QCheckBox("Check for updates at startup")
        self.check_update_btn = QToolButton()
        self.check_update_btn.setText("Check now")
        urow.addWidget(self.auto_update_check)
        urow.addStretch(1)
        urow.addWidget(self.check_update_btn)
        s.addLayout(urow)
        self.update_hint = muted("")
        s.addStretch(1)

        foot = QVBoxLayout()
        foot.setContentsMargins(14, 8, 14, 14)
        foot.setSpacing(6)
        self.export_btn = QPushButton("Export selected")
        self.export_btn.setObjectName("primary")
        self.export_all_btn = QPushButton("Export all")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.hide()
        self.start_now_btn = QPushButton("Start scheduled now")
        self.start_now_btn.setToolTip("Run the scheduled videos now, ignoring the time window until the queue is done")
        self.start_now_btn.hide()
        foot.addWidget(self.export_btn)
        foot.addWidget(self.start_now_btn)
        r2 = QHBoxLayout()
        r2.addWidget(self.export_all_btn, 1)
        r2.addWidget(self.stop_btn, 1)
        foot.addLayout(r2)
        spl.addLayout(foot)
        sp.setMinimumWidth(380)
        sp.setMaximumWidth(460)
        body.addWidget(sp)
        body.setSizes([280, 880, 390])
        body.setStretchFactor(1, 1)

        # ---- status bar
        sb = self.statusBar()
        self.status_text = QLabel("Ready")
        self.status_text.setObjectName("muted")
        self.status_bar = QProgressBar()
        self.status_bar.setRange(0, 1000)
        self.status_bar.setTextVisible(False)
        self.status_bar.setFixedWidth(260)
        self.status_bar.setFixedHeight(6)
        self.status_bar.hide()
        self.share_text = QLabel("")
        self.share_text.setObjectName("muted")
        sb.addWidget(self.status_text, 1)
        sb.addPermanentWidget(self.share_text)
        sb.addPermanentWidget(self.status_bar)

        # ---- log dock
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setWidget(self.log_view)
        self.log_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()

        self.play_timer = QTimer(self)
        self.play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.play_timer.timeout.connect(self._play_tick)

    def _connect(self):
        b = self.bridge
        b.engine_ready.connect(self._engine_ready)
        b.engine_failed.connect(self._engine_failed)
        b.item_loaded.connect(self._item_loaded)
        b.item_changed.connect(self._item_changed)
        b.progress.connect(self._on_progress)
        b.log.connect(self._append_log)
        b.frame_ready.connect(self._frame_ready)
        b.ai_ready.connect(self._ai_ready)
        b.export_idle.connect(self._export_idle)
        b.thumbs_ready.connect(self._thumbs_ready)
        b.clip_progress.connect(self._clip_progress)
        b.clip_done.connect(self._clip_done)
        b.warm_done.connect(self._warm_done)

        self.add_btn.clicked.connect(self._add_dialog)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn.clicked.connect(self._clear_done)
        self.queue_list.currentRowChanged.connect(self._select_row)
        self.queue_list.itemSelectionChanged.connect(self._update_buttons)
        self.log_btn.toggled.connect(self.log_dock.setVisible)
        self.node_mode_btn.toggled.connect(self._toggle_node_mode)
        self.add_node_btn.clicked.connect(self._add_node_dialog)
        self.bridge.nodes_status.connect(self._nodes_status)
        self.log_dock.visibilityChanged.connect(self.log_btn.setChecked)
        self.mode_group.buttonClicked.connect(lambda btn: self.view.set_mode(btn.text()))
        for _, (btn, z) in self.zoom_btns.items():
            btn.clicked.connect(lambda _=False, z=z: self.view.set_zoom(z))
        self.view.on_zoom_changed = self._zoom_changed

        self.frame_timer = QTimer(self)
        self.frame_timer.setSingleShot(True)
        self.frame_timer.setInterval(120)
        self.frame_timer.timeout.connect(self._load_original_frame)
        self.timeline.playheadMoved.connect(self._playhead_moved)
        self.timeline.rangeChanged.connect(self._range_changed)
        self.to_in_btn.clicked.connect(lambda: self._seek(self.current.r_in if self.current else 0))
        self.to_out_btn.clicked.connect(lambda: self._seek(self.current.r_out - 1 / self._fps() if self.current else 0))
        self.step_back_btn.clicked.connect(lambda: self._step(-1))
        self.step_fwd_btn.clicked.connect(lambda: self._step(1))
        self.play_btn.clicked.connect(self._toggle_play)
        self.set_in_btn.clicked.connect(self._set_in)
        self.set_out_btn.clicked.connect(self._set_out)
        self.len_box.activated.connect(self._length_chosen)
        self.preview_btn.clicked.connect(self._render_clip)
        self.frame_btn.clicked.connect(self._render_ai_frame)

        self.model_box.currentIndexChanged.connect(self._model_changed)
        self.add_model_btn.clicked.connect(self._add_model_dialog)
        self.library_btn.clicked.connect(self._open_library)
        self.bridge.catalog_checked.connect(self._catalog_checked)
        self.remove_model_btn.clicked.connect(self._remove_model)
        self.models_folder_btn.clicked.connect(self._open_models_folder)
        self.find_models_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://openmodeldb.info/?t=arch%3Aspan+arch%3Acompact")))
        self.trt_retry_btn.clicked.connect(self._retry_trt)
        self.bridge.models_changed.connect(self._models_changed)
        self.trt_check.toggled.connect(self._trt_changed)
        self.cq_slider.valueChanged.connect(self._cq_changed)
        self.folder_btn.clicked.connect(self._pick_folder)
        self.export_btn.clicked.connect(lambda: self._export(selected_only=True))
        self.export_all_btn.clicked.connect(lambda: self._export(selected_only=False))
        self.stop_btn.clicked.connect(self._stop)
        self.start_now_btn.clicked.connect(self._start_scheduled_now)
        self.startup_check.toggled.connect(self._startup_toggled)
        self.tray_check.toggled.connect(lambda on: self.settings.setValue("close_to_tray", "true" if on else "false"))
        self.bridge.audio_analyzed.connect(self._audio_analyzed)
        self.bridge.audio_model_progress.connect(self._audio_model_progress)
        self.audio_model_btn.clicked.connect(self._download_audio_model)
        self.upscale_mode_box.currentIndexChanged.connect(self._audio_settings_changed)
        self.audio_quality_box.currentIndexChanged.connect(self._audio_settings_changed)
        self.audio_format_box.currentIndexChanged.connect(self._audio_settings_changed)
        self.update_btn.clicked.connect(self._show_update_dialog)
        self.version_btn.clicked.connect(lambda: self._check_updates(manual=True))
        self.check_update_btn.clicked.connect(lambda: self._check_updates(manual=True))
        self.auto_update_check.toggled.connect(
            lambda on: self.settings.setValue("auto_update", "true" if on else "false"))
        self.bridge.update_checked.connect(self._update_checked)
        self.bridge.update_done.connect(self._update_done)
        self.sched_check.toggled.connect(self._sched_settings_changed)
        self.sched_start.timeChanged.connect(self._sched_settings_changed)
        self.sched_end.timeChanged.connect(self._sched_settings_changed)
        self.sched_timer = QTimer(self)
        self.sched_timer.setInterval(10_000)
        self.sched_timer.timeout.connect(self._sched_tick)
        self.sched_timer.start()

        for key, fn in (("Space", self._toggle_play), ("Left", lambda: self._step(-1)),
                        ("Right", lambda: self._step(1)), ("I", self._set_in), ("O", self._set_out),
                        ("P", self._render_clip), ("F", self._render_ai_frame),
                        ("Home", lambda: self._seek(self.current.r_in) if self.current else None),
                        ("End", lambda: self._seek(self.current.r_out - 1 / self._fps()) if self.current else None),
                        ("Ctrl+O", self._add_dialog)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(fn)

    def _set_enabled(self, on):
        for w in (self.add_btn, self.export_btn, self.export_all_btn, self.preview_btn, self.frame_btn,
                  self.model_box, self.trt_check, self.encoder_box):
            w.setEnabled(on)

    # ================================================================ engine startup
    def _load_engine(self):
        try:
            from upscaler import engine, models, trt_backend
            gpus = engine.list_gpus()
            ffmpeg = engine.find_tool("ffmpeg")
            specs = models.all_models(log=lambda m: self.bridge.log.emit(m))
            self.bridge.engine_ready.emit((engine, models, trt_backend, gpus, ffmpeg, specs))
        except Exception as e:  # noqa: BLE001
            self.bridge.engine_failed.emit(str(e))

    def _engine_failed(self, msg):
        self.log("Could not start the AI engine: " + msg)
        QMessageBox.critical(self, APP_NAME, f"Could not start the AI engine:\n\n{msg}")

    def _engine_ready(self, payload):
        engine, models, trt_backend, gpus, ffmpeg, specs = payload
        self.engine, self.models, self.trt = engine, models, trt_backend
        self.previewer = engine.FramePreviewer(log=lambda m: self.bridge.log.emit(m))
        st = self.settings

        self._fill_models(specs, st.value("model", models.MODELS[0].key))

        self.encoder_box.addItems(list(engine.ENCODERS))
        enc = st.value("encoder", "")
        if enc in engine.ENCODERS:
            self.encoder_box.setCurrentText(enc)
        self.container_box.setCurrentText(st.value("container", "mp4"))
        self.cq_slider.setValue(int(st.value("cq", 18)))
        self._cq_changed(self.cq_slider.value())
        self.folder_edit.setText(st.value("folder", ""))
        from upscaler import audio_job
        for w in (self.upscale_mode_box, self.audio_quality_box, self.audio_format_box):
            w.blockSignals(True)
        self.audio_format_box.addItems(list(audio_job.AUDIO_FORMATS))
        self.upscale_mode_box.setCurrentIndex(int(st.value("upscale_mode", 0)))
        self.audio_quality_box.setCurrentIndex(int(st.value("audio_quality", 0)))
        fmt = st.value("audio_format", "")
        if fmt in audio_job.AUDIO_FORMATS:
            self.audio_format_box.setCurrentText(fmt)
        for w in (self.upscale_mode_box, self.audio_quality_box, self.audio_format_box):
            w.blockSignals(False)
        for te, key, default in ((self.sched_start, "sched_start", "23:00"), (self.sched_end, "sched_end", "07:00")):
            te.blockSignals(True)
            te.setTime(QTime.fromString(st.value(key, default), "HH:mm"))
            te.blockSignals(False)
        self.sched_check.blockSignals(True)
        self.sched_check.setChecked(st.value("sched_on", "false") == "true")
        self.sched_check.blockSignals(False)
        self._sched_refresh()
        self.len_box.setCurrentIndex(int(st.value("preview_len_idx", 2)))

        saved_gpus = st.value("gpus", None)
        for gid, name in gpus:
            cb = QCheckBox(f"GPU {gid}  ·  {name.replace('NVIDIA ', '')}")
            cb.setChecked(saved_gpus is None or str(gid) in str(saved_gpus).split(","))
            cb.toggled.connect(lambda _=False: (self._update_trt_hint(), self._discard_clip(), self._warm_up(),
                                                self._update_audio_hint()))
            self.gpu_box.addWidget(cb)
            self.gpu_checks[gid] = cb
        if not gpus:
            lbl = QLabel("No NVIDIA CUDA GPU found.")
            lbl.setStyleSheet(f"color: {C_ERR};")
            self.gpu_box.addWidget(lbl)
        self.gpu_badge.setText(f"{len(gpus)} GPU{'s' if len(gpus) != 1 else ''}")
        self.gpu_badge.setToolTip("\n".join(n for _, n in gpus))

        self.trt_check.blockSignals(True)
        self.trt_check.setChecked(trt_backend.available() and st.value("tensorrt", "true") == "true")
        self.trt_check.blockSignals(False)
        self.trt_check.setToolTip(trt_backend.status())
        self._trt_changed(self.trt_check.isChecked())

        self.log(f"Ready. GPUs: {', '.join(n for _, n in gpus) or 'none'}. {trt_backend.status()}. ffmpeg: {ffmpeg}")
        self._set_enabled(bool(gpus))
        self.trt_check.setEnabled(bool(gpus) and trt_backend.available())
        for it in self.items:
            if it.status == "Loading":
                threading.Thread(target=self._probe_item, args=(it,), daemon=True).start()
        self._update_buttons()
        self._check_nodes()
        self.auto_update_check.blockSignals(True)
        self.auto_update_check.setChecked(st.value("auto_update", "true") == "true")
        self.auto_update_check.blockSignals(False)
        if self.auto_update_check.isChecked():
            self._check_updates(manual=False)
        else:
            self.update_hint.setText("")
        self._check_catalog()
        for cb, val in ((self.startup_check, startup_enabled()),
                        (self.tray_check, st.value("close_to_tray", "true") == "true")):
            cb.blockSignals(True)
            cb.setChecked(val)
            cb.blockSignals(False)
        if not self.tray:
            self.tray_check.setEnabled(False)
            self.tray_check.setToolTip("The system tray isn't available.")
        self._restore_queue()

    # ================================================================ model library (OpenModelDB)
    def _check_catalog(self):
        """Refresh the model catalogue in the background (at most daily) and count new models / model updates."""
        import json as _json
        try:
            seen = set(_json.loads(self.settings.value("catalog_seen", "[]")))
        except (TypeError, ValueError):
            seen = set()

        def work():
            from upscaler import catalog
            try:
                cat = [m for m in catalog.load_catalog() if m.supported]
            except Exception:  # noqa: BLE001
                return                                    # offline: the library still opens from cache
            new = len({m.id for m in cat} - seen) if seen else 0
            inst, imap = catalog.installed_ids(), catalog._installed_map()
            upd = sum(1 for m in cat if catalog.status_of(m, inst, imap) == "update")
            self.bridge.catalog_checked.emit(new, upd)
        threading.Thread(target=work, daemon=True).start()

    def _catalog_checked(self, new, upd):
        parts = ([f"{new} new"] if new else []) + ([f"{upd} update{'s' if upd > 1 else ''}"] if upd else [])
        self.library_btn.setText("Browse model library" + (f"  ({', '.join(parts)})" if parts else ""))
        if parts:
            self.log(f"Model library: {', '.join(parts)}.")

    def _open_library(self):
        from model_library import ModelLibrary

        def installed(spec):
            if self.exporting:
                return
            self.engine.release_models()                 # an updated model must be reloaded
            self._models_changed(spec.key, None)
        dlg = ModelLibrary(self, self.settings, installed)
        dlg.exec()
        self._check_catalog()                            # refresh the "(N new)" count

    def _audio_settings_changed(self, *_):
        if not self.engine:
            return
        self.settings.setValue("upscale_mode", self.upscale_mode_box.currentIndex())
        self.settings.setValue("audio_quality", self.audio_quality_box.currentIndex())
        self.settings.setValue("audio_format", self.audio_format_box.currentText())
        self._update_audio_hint()
        if self.current and self.current.kind == "audio":
            self.out_res_sub.setText(f"Saved as {self.audio_format_box.currentText()}")

    # ================================================================ updates (git)
    def _check_updates(self, manual):
        from upscaler import updater
        reason = updater.status()
        if reason:
            self.update_hint.setText(reason.splitlines()[0])
            if manual:
                QMessageBox.information(self, APP_NAME, reason)
            return
        self.update_hint.setText("Checking GitHub for updates...")
        self.check_update_btn.setEnabled(False)

        def work():
            try:
                self.bridge.update_checked.emit(updater.check_latest(), None, manual)
            except Exception as e:  # noqa: BLE001
                self.bridge.update_checked.emit(None, str(e), manual)
        threading.Thread(target=work, daemon=True).start()

    def _update_checked(self, release, error, manual):
        from upscaler import __version__ as muss_version
        self.check_update_btn.setEnabled(True)
        self.pending_update = release
        if error:
            self.update_hint.setText("Couldn't check for updates (offline?).")
            self.log("Update check failed: " + error)
            if manual:
                QMessageBox.warning(self, APP_NAME, f"Couldn't check for updates:\n\n{error}")
            return
        if release:
            self.update_btn.setText(f"Update to v{release['version']}")
            self.update_btn.show()
            self.update_hint.setText(f"MUSS v{release['version']} is available (you have v{muss_version}).")
            self.log(f"Update available: v{release['version']}")
            if manual:
                self._show_update_dialog()
        else:
            self.update_btn.hide()
            self.update_hint.setText(f"You have the latest version (v{muss_version}).")
            if manual:
                QMessageBox.information(self, APP_NAME, f"MUSS v{muss_version} is the latest version.")

    def _show_update_dialog(self):
        from upscaler import __version__ as muss_version
        rel = getattr(self, "pending_update", None)
        if not rel:
            return
        if self.exporting or self.scheduled:
            QMessageBox.information(self, APP_NAME, "Finish or stop the export queue (including scheduled "
                                    "videos) before updating; MUSS restarts after the update.")
            return
        box = QMessageBox(self)
        box.setWindowTitle(f"{APP_NAME} update")
        box.setText(f"MUSS v{rel['version']} is available (you have v{muss_version}).")
        box.setInformativeText("What's new:\n" + rel["notes"][:1500] +
                               "\n\nMUSS will download the update from GitHub and restart.")
        upd = box.addButton("Update and restart", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is not upd:
            return
        self._discard_clip()
        self.update_btn.setEnabled(False)
        self.update_btn.setText("Updating...")
        self.status_text.setText(f"Updating to v{rel['version']}...")

        def work():
            from upscaler import updater
            try:
                v = updater.apply_update(rel, log=lambda m: self.bridge.log.emit(m))
                self.bridge.update_done.emit(v, None)
            except Exception as e:  # noqa: BLE001
                self.bridge.update_done.emit(None, str(e))
        threading.Thread(target=work, daemon=True).start()

    def _update_done(self, version, error):
        from upscaler import updater
        if error:
            self.update_btn.setEnabled(True)
            self.update_btn.setText(f"Update to v{self.pending_update['version']}")
            self.status_text.setText("Update failed - see Log")
            QMessageBox.warning(self, APP_NAME, f"The update was not applied:\n\n{error}")
            return
        self.status_text.setText(f"Updated to v{version} - restarting...")
        updater.restart_app()
        self._restarting = True
        self.close()

    # ================================================================ settings
    def _spec(self):
        if not self.models:
            return None
        return getattr(self, "specs", {}).get(self.model_box.currentData())

    # ---- model list / custom models
    def _fill_models(self, specs, select_key):
        self.specs = {s.key: s for s in specs}
        self.model_box.blockSignals(True)
        self.model_box.clear()
        builtin = [s for s in specs if s.builtin]
        custom = [s for s in specs if not s.builtin]
        for s in builtin:
            self.model_box.addItem(s.label, s.key)
        if custom:
            self.model_box.insertSeparator(self.model_box.count())
            for s in custom:
                self.model_box.addItem(f"{s.label}   ({s.arch} {s.scale}×)", s.key)
        idx = self.model_box.findData(select_key)
        self.model_box.setCurrentIndex(idx if idx >= 0 else 0)
        self.model_box.blockSignals(False)
        spec = self._spec()
        self.model_desc.setText(spec.description if spec else "")
        self.remove_model_btn.setEnabled(bool(spec and not spec.builtin))
        self.model_box.setToolTip(spec.path or spec.filename if spec else "")

    def _add_model_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add upscaling model", self.settings.value("last_model_dir", os.path.expanduser("~\\Downloads")),
            "Model weights (*.pth *.safetensors *.ckpt *.bin);;All files (*.*)")
        if files:
            self.settings.setValue("last_model_dir", os.path.dirname(files[0]))
            self.add_models(files)

    def add_models(self, files):
        if not self.models:
            return
        self.status_text.setText(f"Checking {len(files)} model file(s)...")

        def work():
            added, errors = None, []
            for f in files:
                try:
                    spec = self.models.add_custom_model(f)
                    added = spec.key
                    self.bridge.log.emit(f"Added model {spec.label}: {spec.description}")
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{os.path.basename(f)}:\n{e}")
            self.bridge.models_changed.emit(added, "\n\n".join(errors) or None)
        threading.Thread(target=work, daemon=True).start()

    def _models_changed(self, select_key, error):
        specs = self.models.all_models(log=self.log)
        self._fill_models(specs, select_key or self.model_box.currentData())
        if select_key:
            self._model_changed()
            self.status_text.setText(f"Model added: {self._spec().label}")
        if error:
            self.status_text.setText("Some model files could not be added")
            QMessageBox.warning(self, APP_NAME, f"These files can't be used:\n\n{error[:1500]}")

    def _remove_model(self):
        spec = self._spec()
        if not spec or spec.builtin:
            return
        if QMessageBox.question(self, APP_NAME, f"Remove the model “{spec.label}”?\n\nThis deletes "
                                f"{os.path.basename(spec.path)} from the app's models folder.") \
                != QMessageBox.StandardButton.Yes:
            return
        if self.exporting:
            QMessageBox.information(self, APP_NAME, "Wait for the export to finish before removing a model.")
            return
        self._discard_clip()
        self.engine.release_models()
        try:
            self.models.remove_custom_model(spec)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"Could not remove the model:\n{e}")
        self._fill_models(self.models.all_models(log=self.log), self.models.MODELS[0].key)
        self._model_changed()

    def _open_models_folder(self):
        if self.models:
            os.makedirs(self.models.CUSTOM_DIR, exist_ok=True)
            os.startfile(self.models.CUSTOM_DIR)

    def _retry_trt(self):
        spec, it = self._spec(), self.current
        if spec and it and it.info:
            ph, pw = self.trt.padded_size(spec, it.info.height, it.info.width)
            try:
                os.remove(self.trt._failed_path(spec, ph, pw))
            except OSError:
                pass
            self._update_trt_hint()

    def _gpus(self):
        return [g for g, cb in self.gpu_checks.items() if cb.isChecked()]

    def _model_changed(self, *_):
        spec = self._spec()
        if not spec:
            return
        self.model_desc.setText(spec.description)
        self.remove_model_btn.setEnabled(not spec.builtin)
        self.model_box.setToolTip(spec.path or spec.filename)
        self.settings.setValue("model", spec.key)
        self._discard_clip()
        self.view.set_after(None)
        self._update_trt_hint()
        self._warm_up()

    def _trt_changed(self, on):
        self.trt_badge.setObjectName("badgeOn" if on else "badge")
        self.trt_badge.setText("TensorRT on" if on else "TensorRT off")
        self.trt_badge.style().unpolish(self.trt_badge)
        self.trt_badge.style().polish(self.trt_badge)
        if self.trt:
            self.settings.setValue("tensorrt", "true" if on else "false")
            self._discard_clip()
            self._warm_up()
        self._update_trt_hint()

    def _use_trt_for_preview(self):
        """TensorRT is used for previews only when its engines already exist (no surprise multi-minute build)."""
        it, spec = self.current, self._spec()
        if not (self.trt_check.isChecked() and it and it.info and spec):
            return False
        return self.engine.trt_engines_ready(spec, self._gpus(), it.info.width, it.info.height)

    def _update_trt_hint(self):
        if not self.trt:
            return
        self.trt_retry_btn.hide()
        if not self.trt.available():
            self.trt_hint.setText("TensorRT is not installed in this environment.")
            return
        if not self.trt_check.isChecked():
            self.trt_hint.setText("Off - using PyTorch.")
            return
        spec, it = self._spec(), self.current
        reason = self.engine.trt_unsupported(spec, it.info.width, it.info.height) if (it and it.info and spec) else None
        if reason:
            self.trt_hint.setText(f"TensorRT couldn't convert this model, so it runs with PyTorch. ({reason[:140]})")
            self.trt_hint.setToolTip(reason)
            self.trt_retry_btn.show()
            return
        if it and it.info and spec:
            ph, pw = self.trt.padded_size(spec, it.info.height, it.info.width)
            missing = [g for g in self._gpus() if not os.path.exists(self.trt.engine_path(spec, g, ph, pw))]
            if missing:
                self.trt_hint.setText(f"Engines for this model at {it.info.width}×{it.info.height} will be built on the "
                                      f"first export (GPU {', '.join(map(str, missing))}; a few minutes, once). "
                                      "Previews use PyTorch until then.")
            else:
                self.trt_hint.setText("Optimized engines are ready for this model and resolution.")
        else:
            self.trt_hint.setText("Compiles each model for each GPU the first time (a few minutes), then runs faster.")

    def _cq_changed(self, v):
        words = "visually lossless" if v <= 16 else "high" if v <= 20 else "medium" if v <= 26 else "small files"
        self.cq_value.setText(f"CQ {v}  ·  {words}")
        if self.engine:
            self.settings.setValue("cq", v)

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.folder_edit.text() or os.path.expanduser("~"))
        if d:
            self.folder_edit.setText(os.path.normpath(d))

    def _zoom_changed(self):
        self.zoom_label.setText(f"{self.view.current_zoom() * 100:.0f}%" if self.view.img_w else "")

    def _warm_up(self):
        """Load the AI model onto the GPUs in the background so the first preview starts quickly."""
        it, spec = self.current, self._spec()
        if not (self.engine and it and it.info and spec and self._gpus()) or self.exporting or self.warming:
            return
        if self.clip and self.clip.job and not self.clip.complete:
            return
        self.warming = True
        gpus, use_trt, w, h = self._gpus(), self._use_trt_for_preview(), it.info.width, it.info.height

        def work():
            try:
                self.engine.get_nets(spec, gpus, use_trt, w, h, log=lambda m: self.bridge.log.emit(m), build_trt=False)
                self.bridge.warm_done.emit(None)
            except Exception as e:  # noqa: BLE001
                self.bridge.warm_done.emit(str(e))
        self.status_text.setText("Loading AI model onto the GPUs...")
        threading.Thread(target=work, daemon=True).start()

    def _warm_done(self, err):
        self.warming = False
        if err:
            self.log("Model warm-up failed: " + err)
        elif not self.exporting:
            self.status_text.setText("AI model ready.")

    # ================================================================ queue
    def _add_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add videos or audio", self.settings.value("last_dir", ""),
            "Videos and audio (" + " ".join("*" + e for e in VIDEO_EXTS + AUDIO_EXTS) + ");;"
            "Videos (" + " ".join("*" + e for e in VIDEO_EXTS) + ");;"
            "Audio (" + " ".join("*" + e for e in AUDIO_EXTS) + ");;All files (*.*)")
        if files:
            self.settings.setValue("last_dir", os.path.dirname(files[0]))
            self.add_files(files)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        self.add_files([p for p in paths if p.lower().endswith(VIDEO_EXTS + AUDIO_EXTS)])
        model_files = [p for p in paths if self.models and p.lower().endswith(self.models.MODEL_EXTS)]
        if model_files:
            self.add_models(model_files)

    def add_files(self, paths):
        known = {os.path.normcase(i.path) for i in self.items}
        for p in paths:
            p = os.path.normpath(p)
            if os.path.normcase(p) in known or not os.path.isfile(p):
                continue
            it = QueueItem(p)
            it.widget = QueueRow(it)
            it.list_item = QListWidgetItem()
            it.list_item.setSizeHint(it.widget.sizeHint())
            self.queue_list.addItem(it.list_item)
            self.queue_list.setItemWidget(it.list_item, it.widget)
            self.items.append(it)
            if self.engine:
                threading.Thread(target=self._probe_item, args=(it,), daemon=True).start()
        if self.queue_list.currentRow() < 0 and self.items:
            self.queue_list.setCurrentRow(0)
        self._update_buttons()
        self._queue_changed()

    def _probe_item(self, it: QueueItem):
        from upscaler import audio_job
        try:
            it.ainfo = audio_job.probe_audio(it.path)
            if it.path.lower().endswith(AUDIO_EXTS):
                if it.ainfo is None:
                    raise RuntimeError("no audio track found")
                it.kind = "audio"
            else:
                it.info = self.engine.probe(it.path)
                buf = self.engine.grab_frame(it.path, it.info, it.info.duration * 0.1)
                it.thumb = rgb_to_image(buf, it.info.width, it.info.height).scaled(
                    192, 108, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            it.status = "Ready"
        except Exception as e:  # noqa: BLE001
            it.status, it.detail = "Failed", f"can't read: {e}"
        self.bridge.item_loaded.emit(it)

    # ---- audio analysis
    def _audio_relevant(self, it):
        return it is not None and it.ainfo is not None and (
            it.kind == "audio" or self.upscale_mode_box.currentIndex() > 0)

    def _analyze_audio(self, it):
        """Measure where this item's treble stops (background, once per item)."""
        if not self._audio_relevant(it) or it.cutoff is not None or it.analyzing:
            return
        it.analyzing = True

        def work():
            from upscaler import audio_job
            try:
                it.cutoff = audio_job.analyze_cutoff(it.path, it.ainfo)
            except Exception as e:  # noqa: BLE001
                self.bridge.log.emit(f"Audio analysis failed for {os.path.basename(it.path)}: {e}")
                it.cutoff = -1
            it.analyzing = False
            self.bridge.audio_analyzed.emit(it)
        threading.Thread(target=work, daemon=True).start()

    def _audio_analyzed(self, it):
        if it is self.current:
            self._update_audio_hint()
            if it.kind == "audio":
                self._show_item(it)

    def _audio_estimate(self, it):
        """Rough processing time for restoring this item's audio on the selected GPUs."""
        from upscaler import audio
        a = it.ainfo
        chans = a.channels - (1 if a.channels in (6, 8) else 0)
        chunks = chans * audio.chunks_per_channel(int(a.duration * audio.SR))
        steps = AUDIO_QUALITY[self.audio_quality_box.currentIndex()][1]
        # measured end to end: ~0.09 chunks/s per GPU at 25 steps (RTX 4070 + 3060 average), plus ~1 min to load
        per_gpu = 0.09 * 25 / steps
        return 60 + chunks / (per_gpu * max(len(self._gpus()), 1))

    def _update_audio_hint(self):
        it = self.current
        if not self.engine:
            return
        from upscaler import audio, audio_job
        lines = []
        if not audio.weights_present():
            lines.append("The AudioSR model (6.2 GB) will be downloaded the first time you restore audio.")
        self.audio_model_btn.setVisible(not audio.weights_present())
        if it is None or it.ainfo is None:
            self.audio_hint.setText("\n".join(lines) or "Select a video or audio file.")
            return
        desc = audio_job.describe(it.ainfo)
        if not self._audio_relevant(it):
            self.audio_hint.setText(f"This video's audio: {desc} - kept as it is.")
            return
        if it.cutoff is None:
            lines.insert(0, f"Audio: {desc} - measuring where the treble stops...")
            self._analyze_audio(it)
        elif it.cutoff < 0:
            lines.insert(0, f"Audio: {desc} - couldn't be analysed.")
        elif it.cutoff >= audio.FULL_RANGE_HZ:
            lines.insert(0, f"Audio: {desc}. The treble already reaches {it.cutoff / 1000:.1f} kHz (full range): "
                            "nothing to restore, so the audio is kept as it is.")
        else:
            est = self._audio_estimate(it)
            lines.insert(0, f"Audio: {desc}. The treble stops at {it.cutoff / 1000:.1f} kHz: AudioSR will restore it "
                            f"up to 24 kHz (≈ {fmt_time(est)} on {len(self._gpus())} GPU(s)).")
        self.audio_hint.setText("\n".join(lines))

    def _download_audio_model(self):
        from upscaler import audio
        if audio.weights_present() or getattr(self, "_audio_dl", False):
            return
        if QMessageBox.question(self, APP_NAME, "Download the AudioSR model now?\n\n6.2 GB from Hugging Face "
                                "(haoheliu/audiosr_basic, Apache-2.0 licence). It's needed once, for restoring audio.") \
                != QMessageBox.StandardButton.Yes:
            return
        self._audio_dl = True
        self.audio_model_btn.setEnabled(False)

        def work():
            try:
                audio.download_weights(progress=lambda d, t: self.bridge.audio_model_progress.emit(d, t, None),
                                       log=lambda m: self.bridge.log.emit(m))
                self.bridge.audio_model_progress.emit(1, 1, "done")
            except Exception as e:  # noqa: BLE001
                self.bridge.audio_model_progress.emit(0, 1, str(e))
        threading.Thread(target=work, daemon=True).start()

    def _audio_model_progress(self, done, total, state):
        if state is None:
            self.audio_model_btn.setText(f"Downloading AudioSR...  {done / 1e9:.1f} / {total / 1e9:.1f} GB")
            return
        self._audio_dl = False
        self.audio_model_btn.setEnabled(True)
        self.audio_model_btn.setText("Download AudioSR model (6.2 GB)")
        if state == "done":
            self.status_text.setText("AudioSR model downloaded.")
        else:
            QMessageBox.warning(self, APP_NAME, f"The AudioSR download failed:\n\n{state}")
        self._update_audio_hint()

    def _make_thumbs(self, it: QueueItem):
        """Filmstrip for the timeline: ~16 small frames spread over the video, grabbed in parallel."""
        info = it.info
        n = 16
        tw = 160
        th = max(2, int(round(tw * info.height / info.width / 2)) * 2)
        times = [info.duration * (i + 0.5) / n for i in range(n)]
        out = [None] * n

        def grab(i):
            try:
                out[i] = self.engine.grab_frame(it.path, info, times[i], size=(tw, th))
            except Exception:  # noqa: BLE001
                pass

        threads = [threading.Thread(target=grab, args=(i,), daemon=True) for i in range(n)]
        for k in range(0, n, 6):
            for t in threads[k:k + 6]:
                t.start()
            for t in threads[k:k + 6]:
                t.join()
        thumbs = [(times[i] - info.duration / n / 2, rgb_to_image(b, tw, th)) for i, b in enumerate(out) if b]
        self.bridge.thumbs_ready.emit(it, thumbs)

    def _thumbs_ready(self, it, thumbs):
        it.thumbs = [(t, QPixmap.fromImage(img)) for t, img in thumbs]
        if it is self.current:
            self.timeline.set_thumbs(it.thumbs)

    def _item_loaded(self, it):
        if it.restore is not None:
            self._apply_restore(it)
        if it.status == "Ready":
            prog = self._resume_info(it)
            if prog and prog[0]:
                it.status = "Stopped"
                it.detail = f"{prog[0] * 100 // max(prog[1], 1)}% saved - export again to resume"
        if it.widget:
            it.widget.refresh()
        if it.info:
            threading.Thread(target=self._make_thumbs, args=(it,), daemon=True).start()
        if it is self.current:
            self._show_item(it)
        elif it.kind == "audio":
            self._analyze_audio(it)
        self._update_buttons()

    def _item_changed(self, it):
        if it.widget:
            it.widget.refresh()
        self._update_buttons()
        self._queue_changed()

    def _select_row(self, row):
        it = self.items[row] if 0 <= row < len(self.items) else None
        if it is self.current:
            return
        self._stop_playback()
        self._discard_clip()
        self.current = it
        self._show_item(it)

    def _show_item(self, it):
        self._update_trt_hint()
        if it is None:
            self.view.busy_text = ""
            self.view.set_frames(None, None)
            self.timeline.set_media(0, 30)
            self.out_res.setText("Select a video")
            self._update_buttons()
            self._update_audio_hint()
            return
        self._update_audio_hint()
        if it.kind == "audio" and it.ainfo:
            from upscaler import audio, audio_job
            a = it.ainfo
            self.view.set_frames(None, None)
            if it.cutoff is None:
                verdict = "Measuring where the treble stops..."
            elif it.cutoff < 0:
                verdict = "Couldn't analyse this file."
            elif it.cutoff >= audio.FULL_RANGE_HZ:
                verdict = f"Treble reaches {it.cutoff / 1000:.1f} kHz: already full range, nothing to restore."
            else:
                verdict = (f"Treble stops at {it.cutoff / 1000:.1f} kHz  →  AudioSR restores it up to 24 kHz.")
            self.view.busy_text = ""
            self.view.placeholder = (f"♪  {os.path.basename(it.path)}\n\n{audio_job.describe(a)}  ·  "
                                     f"{fmt_time(a.duration)}\n\n{verdict}")
            self.view.update()
            self.timeline.set_media(0, 30)
            self.out_res.setText(f"Audio  →  48 kHz, full range")
            self.out_res_sub.setText(f"Saved as {self.audio_format_box.currentText()}")
            self.tc_label.setText("")
            self.dur_label.setText("")
            self._update_buttons()
            return
        self.view.placeholder = "Drop videos here or click  + Add videos"
        if not it.info:
            self.view.busy_text = "Reading video..." if it.status == "Loading" else it.detail
            self.view.set_frames(None, None)
            self.timeline.set_media(0, 30)
            return
        i = it.info
        self.out_res.setText(f"{i.width}×{i.height}  →  {i.width * 2}×{i.height * 2}")
        label = "4K UHD" if (i.width * 2, i.height * 2) == (3840, 2160) else "2× upscale"
        self.out_res_sub.setText(f"{label}  ·  {i.frames} frames  ·  {float(i.fps):.3f} fps  ·  audio: {i.audio_codec or 'none'}")
        if it.playhead < 0:
            it.playhead = round(i.duration * 0.1 * float(i.fps)) / float(i.fps)
            it.r_in = it.playhead
            it.r_out = min(it.r_in + self._preview_len(), i.duration)
        self.timeline.set_media(i.duration, float(i.fps))
        self.timeline.set_thumbs(it.thumbs)
        self.timeline.set_range(it.r_in, it.r_out)
        self.timeline.set_playhead(it.playhead)
        self.dur_label.setText("/ " + fmt_tc(i.duration, float(i.fps)))
        self._update_range_label()
        self._update_tc()
        self._load_original_frame()
        self._update_buttons()
        self._warm_up()

    def _remove_selected(self):
        rows = sorted({self.queue_list.row(li) for li in self.queue_list.selectedItems()}, reverse=True)
        for r in rows:
            it = self.items[r]
            if it.status in ("Processing", "Queued", "Scheduled"):  # running work can't be removed
                continue
            self.queue_list.takeItem(r)
            self.items.pop(r)
        self.current = None
        self._select_row(self.queue_list.currentRow())
        self._update_buttons()
        self._queue_changed()

    def _clear_done(self):
        for r in reversed(range(len(self.items))):
            if self.items[r].status == "Done":
                self.queue_list.takeItem(r)
                self.items.pop(r)
        self.current = None
        self._select_row(self.queue_list.currentRow())
        self._update_buttons()
        self._queue_changed()

    def _update_buttons(self):
        ready = self.engine is not None and bool(self.gpu_checks)
        exportable = [i for i in self.items if i.status in ("Ready", "Done", "Failed", "Cancelled", "Stopped")]
        busy = self.exporting
        has_media = self.current is not None and self.current.info is not None
        self.export_btn.setEnabled(ready and bool(self.queue_list.selectedItems()))
        self.export_all_btn.setEnabled(ready and bool(exportable))
        sched = bool(getattr(self, "scheduled", []))
        self.stop_btn.setVisible(busy or sched)
        self.stop_btn.setText("Stop" if busy else "Cancel schedule")
        self.export_all_btn.setVisible(not (busy or sched))
        self.start_now_btn.setVisible(sched and not busy)
        for b in (self.preview_btn, self.frame_btn):
            b.setEnabled(ready and not busy and has_media)
            b.setToolTip("AI previews are paused while exporting." if busy else b.toolTip())
        for b in (self.play_btn, self.step_back_btn, self.step_fwd_btn, self.to_in_btn, self.to_out_btn,
                  self.set_in_btn, self.set_out_btn, self.len_box):
            b.setEnabled(has_media)
        self.queue_hint.setVisible(not self.items)

    # ================================================================ timeline + range
    def _fps(self):
        return float(self.current.info.fps) if self.current and self.current.info else 30.0

    def _preview_len(self):
        _, n = PREVIEW_LENGTHS[self.len_box.currentIndex()]
        return n or 3

    def _update_tc(self):
        if self.current and self.current.info:
            self.tc_label.setText(fmt_tc(self.current.playhead, self._fps()))

    def _update_range_label(self):
        it = self.current
        if not (it and it.info):
            self.range_label.setText("")
            return
        n = max(int(round((it.r_out - it.r_in) * self._fps())), 1)
        gb = n * it.info.width * it.info.height * 3 * 5 / 1e9     # original + 4x-size AI frame
        self.range_label.setText(f"{n} frames · {gb:.1f} GB")
        self.range_label.setToolTip(f"Preview range {it.r_out - it.r_in:.2f} s = {n} frames; "
                                    f"about {gb:.1f} GB of RAM while the preview is kept for playback.")

    def _playhead_moved(self, t, final):
        it = self.current
        if not (it and it.info):
            return
        if self.playing and not final:
            self._stop_playback()
        it.playhead = t
        self._update_tc()
        if not self._show_clip_frame():
            self.frame_timer.start()

    def _seek(self, t):
        it = self.current
        if not (it and it.info):
            return
        t = min(max(round(t * self._fps()) / self._fps(), 0.0), max(it.info.duration - 1 / self._fps(), 0.0))
        it.playhead = t
        self.timeline.set_playhead(t)
        self._update_tc()
        if not self._show_clip_frame():
            self.frame_timer.start()

    def _step(self, n):
        if self.current and self.current.info:
            self._stop_playback()
            self._seek(self.current.playhead + n / self._fps())

    def _range_changed(self, a, b):
        it = self.current
        if not it:
            return
        it.r_in, it.r_out = a, b
        self.len_box.setCurrentIndex(len(PREVIEW_LENGTHS) - 1)      # "Custom"
        self._update_range_label()

    def _set_in(self):
        it = self.current
        if not (it and it.info):
            return
        length = it.r_out - it.r_in
        it.r_in = it.playhead
        it.r_out = min(it.r_in + length, it.info.duration) if it.r_out <= it.r_in else it.r_out
        if it.r_out <= it.r_in:
            it.r_out = min(it.r_in + self._preview_len(), it.info.duration)
        self.timeline.set_range(it.r_in, it.r_out)
        it.r_in, it.r_out = self.timeline.r_in, self.timeline.r_out
        self._update_range_label()

    def _set_out(self):
        it = self.current
        if not (it and it.info):
            return
        it.r_out = it.playhead + 1 / self._fps()
        if it.r_out <= it.r_in:
            it.r_in = max(it.r_out - self._preview_len(), 0.0)
        self.timeline.set_range(it.r_in, it.r_out)
        it.r_in, it.r_out = self.timeline.r_in, self.timeline.r_out
        self.len_box.setCurrentIndex(len(PREVIEW_LENGTHS) - 1)
        self._update_range_label()

    def _length_chosen(self, idx):
        self.settings.setValue("preview_len_idx", idx)
        it = self.current
        _, n = PREVIEW_LENGTHS[idx]
        if not (it and it.info) or not n:
            return
        it.r_in = it.playhead
        it.r_out = min(it.r_in + n, it.info.duration)
        if it.r_out - it.r_in < n:      # near the end: shift the range back
            it.r_in = max(it.r_out - n, 0.0)
        self.timeline.set_range(it.r_in, it.r_out)
        it.r_in, it.r_out = self.timeline.r_in, self.timeline.r_out
        self._update_range_label()

    # ================================================================ original frames + single AI frame
    def _load_original_frame(self):
        it = self.current
        if not it or not it.info:
            return
        fps = self._fps()
        k = int(round(it.playhead * fps))
        self.preview_req += 1
        req = self.preview_req
        key = (it.path, k)
        if key in self.frame_cache:
            img, buf = self.frame_cache[key]
            self._frame_ready((req, it, k, img, buf))
            return
        self.view.busy_text = "Loading frame..." if self.view.before is None else ""

        def work():
            try:
                buf = self.engine.grab_frame(it.path, it.info, k / fps)
                self.bridge.frame_ready.emit((req, it, k, rgb_to_image(buf, it.info.width, it.info.height), buf))
            except Exception as e:  # noqa: BLE001
                self.bridge.log.emit(f"Preview frame failed: {e}")
        threading.Thread(target=work, daemon=True).start()

    def _frame_ready(self, payload):
        req, it, k, img, buf = payload
        if len(self.frame_cache) > 24:
            self.frame_cache.pop(next(iter(self.frame_cache)))
        self.frame_cache[(it.path, k)] = (img, buf)
        if req != self.preview_req or it is not self.current or self._clip_index() is not None:
            return
        self.view.busy_text = ""
        self.preview_buf = buf
        self._shown_clip_frame = None
        self.view.set_frames(img, None, it.info.width * 2, it.info.height * 2)

    def _render_ai_frame(self):
        it = self.current
        if not it or not it.info or not self.frame_btn.isEnabled() or not self.preview_buf:
            return
        if self._clip_index() is not None:
            return      # this frame is already rendered in the preview clip
        gpus = self._gpus() or list(self.gpu_checks)
        spec, gid, use_trt = self._spec(), gpus[0], self._use_trt_for_preview()
        req, buf = self.preview_req, self.preview_buf
        self.view.busy_text = f"Rendering AI frame on GPU {gid}..."
        self.view.update()

        def work():
            try:
                t0 = time.time()
                out = self.previewer.render(spec, gid, buf, it.info.width, it.info.height, use_tensorrt=use_trt)
                img = rgb_to_image(out, it.info.width * 2, it.info.height * 2)
                self.bridge.ai_ready.emit((req, it, img, time.time() - t0, None))
            except Exception as e:  # noqa: BLE001
                self.bridge.ai_ready.emit((req, it, None, 0, str(e)))
        threading.Thread(target=work, daemon=True).start()

    def _ai_ready(self, payload):
        req, it, img, dt, err = payload
        if req != self.preview_req or it is not self.current:
            return
        self.view.busy_text = ""
        if err:
            self.log("AI preview failed: " + err)
            self.view.update()
            return
        self.view.set_after(img)
        self.status_text.setText(f"AI frame rendered in {dt:.2f} s")

    # ================================================================ preview clip (range render + playback)
    def _clip_index(self):
        c, it = self.clip, self.current
        if not (c and it and c.path == it.path):
            return None
        return c.index_at(it.playhead)

    def _show_clip_frame(self):
        """If the playhead is inside the rendered preview, show that frame from memory. Returns True if shown."""
        k = self._clip_index()
        if k is None:
            return False
        if k != self._shown_clip_frame:
            c = self.clip
            before = wrap_image(c.src[k], c.w, c.h)
            after = wrap_image(c.ai[k], c.w * 2, c.h * 2)
            self.view.busy_text = ""
            self.view.set_frames(before, after, c.w * 2, c.h * 2, keep=(c.src[k], c.ai[k]))
            self.preview_buf = c.src[k]
            self._shown_clip_frame = k
        return True

    def _discard_clip(self):
        c = self.clip
        if c and c.job and not c.complete:
            c.job.cancel()
        self.clip = None
        self._shown_clip_frame = None
        self.timeline.set_rendered(None)
        self.timeline.set_rendering(None)
        self._stop_playback()

    def _render_clip(self):
        it = self.current
        if not (it and it.info) or not self.preview_btn.isEnabled():
            return
        spec, gpus = self._spec(), self._gpus()
        if not gpus:
            QMessageBox.warning(self, APP_NAME, "Select at least one GPU.")
            return
        fps = self._fps()
        n = max(int(round((it.r_out - it.r_in) * fps)), 1)
        per_frame = it.info.width * it.info.height * 3 * 5
        budget = int(self.engine.free_ram_bytes() * 0.5)
        if n * per_frame > budget:
            n_ok = max(budget // per_frame, 1)
            QMessageBox.information(
                self, APP_NAME,
                f"This preview range needs about {n * per_frame / 1e9:.1f} GB of memory, but only "
                f"{budget / 1e9:.1f} GB can be used right now.\nThe preview will cover the first "
                f"{n_ok / fps:.1f} seconds of the range.")
            n = n_ok
        self._discard_clip()
        use_trt = self._use_trt_for_preview()
        clip = PreviewClip(it.path, spec.key, it.r_in, fps, it.info.width, it.info.height, n)
        self.clip = clip
        self.timeline.set_rendering((clip.start, clip.end))
        self._seek(clip.start)
        self.view.busy_text = "Rendering preview..."
        self.view.update()
        self.status_text.setText(f"Rendering {n}-frame preview with {spec.label}"
                                 f"{' (TensorRT)' if use_trt else ''} on {len(gpus)} GPU(s)...")
        last_emit = [0.0]

        def sink(idx, src, out):
            clip.src.append(src)
            clip.ai.append(out)
            now = time.time()
            if idx == 0 or now - last_emit[0] > 0.1:
                last_emit[0] = now
                self.bridge.clip_progress.emit(clip)

        job = self.engine.UpscaleJob(
            it.path, None, spec, gpus, use_tensorrt=use_trt, start=clip.start, max_frames=n,
            frame_sink=sink, on_log=lambda m: self.bridge.log.emit(m))
        clip.job = job

        def work():
            t0 = time.time()
            try:
                job.run()
                self.bridge.clip_done.emit(clip, (None, time.time() - t0))
            except Exception as e:  # noqa: BLE001
                self.bridge.clip_done.emit(clip, (str(e), 0))
        threading.Thread(target=work, daemon=True).start()
        self._start_playback()          # plays frames as soon as they arrive, like a buffering video

    def _clip_progress(self, clip):
        if clip is not self.clip:
            return
        self.timeline.set_rendered((clip.start, clip.time_of(len(clip.ai))))
        if self.view.busy_text.startswith("Rendering preview"):
            self.view.busy_text = f"Rendering preview  {len(clip.ai)}/{clip.total}"
        if not self.playing:
            self._show_clip_frame()

    def _clip_done(self, clip, result):
        err, dt = result
        if clip is not self.clip:
            return
        clip.complete = True
        self.view.busy_text = ""
        self.timeline.set_rendering(None)
        self.timeline.set_rendered((clip.start, clip.time_of(len(clip.ai))) if clip.ai else None)
        if err:
            self.log("Preview failed: " + err)
            self.status_text.setText("Preview failed - see Log")
        elif clip.ai:
            self.status_text.setText(f"Preview ready: {len(clip.ai)} frames in {dt:.1f} s "
                                     f"({len(clip.ai) / max(dt, 1e-6):.1f} fps). Space to play, arrows to step.")
        self._show_clip_frame()
        self.view.update()

    def _toggle_play(self):
        if self.playing:
            self._stop_playback()
        elif self.clip and self.current and self.clip.path == self.current.path and self.clip.ai:
            if self._clip_index() is None:
                self._seek(self.clip.start)
            self._start_playback()
        else:
            self._render_clip()

    def _start_playback(self):
        if not self.clip:
            return
        self.playing = True
        self.view.set_fast(True)
        self.play_btn.setText("❚❚")
        self.play_timer.start(max(int(1000 / self.clip.fps), 10))

    def _stop_playback(self):
        self.playing = False
        self.play_timer.stop()
        self.view.set_fast(False)
        self.play_btn.setText("▶")

    def _play_tick(self):
        c, it = self.clip, self.current
        if not (c and it and c.path == it.path):
            self._stop_playback()
            return
        k = c.index_at(it.playhead)
        if k is None:
            k = -1
        nxt = k + 1
        if nxt >= c.total or (c.complete and nxt >= len(c.ai)):
            nxt = 0                                  # loop the preview range
        if nxt >= len(c.ai):
            return                                   # still rendering: wait for the next frame
        if nxt == 0 and k == -1 and not c.ai:
            return
        it.playhead = c.time_of(nxt)
        self.timeline.set_playhead(it.playhead)
        self._update_tc()
        self._show_clip_frame()

    # ================================================================ export
    def _snapshot_settings(self):
        gpus = self._gpus()
        st = self.settings
        st.setValue("gpus", ",".join(map(str, gpus)))
        st.setValue("encoder", self.encoder_box.currentText())
        st.setValue("container", self.container_box.currentText())
        st.setValue("folder", self.folder_edit.text())
        return {
            "spec": self._spec(), "gpus": gpus, "encoder": self.encoder_box.currentText(),
            "cq": self.cq_slider.value(), "ext": self.container_box.currentText(),
            "folder": self.folder_edit.text().strip(), "test": self.test_check.isChecked(),
            "trt": self.trt_check.isChecked(), "range": self.range_check.isChecked(),
            "nodes": [n for n in self._load_nodes() if n.get("enabled", True)],
            "upscale_mode": self.upscale_mode_box.currentIndex(),        # 0 video, 1 video+audio, 2 audio only
            "audio_steps": AUDIO_QUALITY[self.audio_quality_box.currentIndex()][1],
            "audio_format": self.audio_format_box.currentText(),
        }

    @staticmethod
    def _output_path(src, s, suffix):
        """First free output name - or the name of an unfinished export of this video, so it resumes."""
        folder = s["folder"] if s["folder"] and os.path.isdir(s["folder"]) else os.path.dirname(src)
        stem = os.path.splitext(os.path.basename(src))[0] + suffix
        path = os.path.join(folder, f"{stem}.{s['ext']}")
        n = 2
        while True:
            if os.path.isdir(path + ".parts"):
                return path                        # unfinished export: resume into the same file
            if not os.path.exists(path) or os.path.normcase(path) == os.path.normcase(src):
                return path
            path = os.path.join(folder, f"{stem} ({n}).{s['ext']}")
            n += 1

    @staticmethod
    def _suffix(s):
        suffix = "_4K"
        if s.get("seconds") and s.get("range"):
            suffix += f"_{int(s['start']) // 60:02d}m{int(s['start']) % 60:02d}s"
        elif s.get("seconds"):
            suffix += "_test"
        return suffix

    def _remote_workers(self, s):
        """One worker per GPU of every enabled, reachable render node (runs in the export thread)."""
        if not s.get("nodes"):
            return []
        from upscaler import node
        return node.remote_workers_for(s["nodes"], log=lambda m: self.bridge.log.emit(m))

    # ================================================================ render nodes
    def _load_nodes(self):
        import json
        try:
            return json.loads(self.settings.value("nodes", "[]"))
        except (TypeError, ValueError):
            return []

    def _save_nodes(self, nodes):
        import json
        self.settings.setValue("nodes", json.dumps(nodes))

    def _refresh_nodes_ui(self, status=None):
        status = status or getattr(self, "_node_status", {})
        while self.nodes_box.count():
            w = self.nodes_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        for n in self._load_nodes():
            key = f"{n['host']}:{n.get('port', 47820)}"
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox(n.get("name") or n["host"])
            cb.setChecked(n.get("enabled", True))
            cb.toggled.connect(lambda on, k=key: self._set_node_enabled(k, on))
            st = status.get(key)
            if isinstance(st, dict):
                text = " · ".join(g["name"].replace("NVIDIA GeForce ", "") for g in st.get("gpus", [])) or "no GPUs"
                lbl = QLabel(f"online · {text}")
                lbl.setStyleSheet(f"color: {C_OK}; font-size: 9pt;")
            elif st is None:
                lbl = QLabel("checking...")
                lbl.setObjectName("muted")
            else:
                lbl = QLabel("offline")
                lbl.setStyleSheet(f"color: {C_ERR}; font-size: 9pt;")
                lbl.setToolTip(str(st))
            rm = QToolButton()
            rm.setText("✕")
            rm.setToolTip(f"Remove {key}")
            rm.clicked.connect(lambda _=False, k=key: self._remove_node(k))
            cb.setToolTip(key)
            rl.addWidget(cb)
            rl.addWidget(lbl, 1)
            rl.addWidget(rm)
            self.nodes_box.addWidget(row)

    def _check_nodes(self):
        nodes = self._load_nodes()
        if not nodes:
            self._refresh_nodes_ui({})
            return
        self._refresh_nodes_ui({})

        def work():
            from upscaler import node
            res = {}
            for n in nodes:
                key = f"{n['host']}:{n.get('port', 47820)}"
                try:
                    res[key] = node.node_info(n["host"], n.get("port", 47820), n["code"])
                except Exception as e:  # noqa: BLE001
                    res[key] = str(e)
            self.bridge.nodes_status.emit(res)
        threading.Thread(target=work, daemon=True).start()

    def _nodes_status(self, res):
        self._node_status = res
        self._refresh_nodes_ui(res)

    def _set_node_enabled(self, key, on):
        nodes = self._load_nodes()
        for n in nodes:
            if f"{n['host']}:{n.get('port', 47820)}" == key:
                n["enabled"] = on
        self._save_nodes(nodes)

    def _remove_node(self, key):
        self._save_nodes([n for n in self._load_nodes() if f"{n['host']}:{n.get('port', 47820)}" != key])
        self._refresh_nodes_ui()

    def _add_node_dialog(self):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout
        from upscaler import node
        dlg = QDialog(self)
        dlg.setWindowTitle("Add render node")
        form = QFormLayout(dlg)
        intro = QLabel("On the other PC, open MUSS and switch on “Render node” in the top bar\n"
                       "(or run Render Node.bat). It shows an address and a pairing code.")
        intro.setObjectName("muted")
        form.addRow(intro)
        addr = QLineEdit()
        addr.setPlaceholderText("192.168.1.50   or   192.168.1.50:47820")
        code = QLineEdit()
        code.setPlaceholderText("8-character pairing code")
        result = QLabel("")
        result.setWordWrap(True)
        form.addRow("Address", addr)
        form.addRow("Pairing code", code)
        form.addRow(result)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Connect and add")
        form.addRow(bb)
        bb.rejected.connect(dlg.reject)

        def accept():
            text = addr.text().strip()
            host, _, port = text.partition(":")
            try:
                port = int(port) if port else node.DEFAULT_PORT
            except ValueError:
                result.setText("The port must be a number.")
                return
            c = code.text().strip().upper()
            result.setText("Connecting...")
            QApplication.processEvents()
            try:
                info = node.node_info(host, port, c, timeout=5)
            except Exception as e:  # noqa: BLE001
                msg = "Wrong pairing code." if "403" in str(e) else f"Can't reach {host}:{port} ({e})."
                result.setText(f"<span style='color:{C_ERR}'>{msg}</span>")
                return
            nodes = [n for n in self._load_nodes() if not (n["host"] == host and n.get("port", 47820) == port)]
            nodes.append({"host": host, "port": port, "code": c, "name": info.get("name") or host, "enabled": True})
            self._save_nodes(nodes)
            self._node_status = {**getattr(self, "_node_status", {}), f"{host}:{port}": info}
            self._refresh_nodes_ui()
            self.log(f"Added render node {info.get('name')} ({host}:{port}) with "
                     f"{', '.join(g['name'] for g in info.get('gpus', []))}.")
            dlg.accept()
        bb.accepted.connect(accept)
        dlg.exec()

    def _toggle_node_mode(self, on):
        from upscaler import node
        if on:
            if self.exporting:
                self.node_mode_btn.setChecked(False)
                QMessageBox.information(self, APP_NAME, "Finish or stop the export first.")
                return
            code = self.settings.value("node_code", "") or node.new_pairing_code()
            self.settings.setValue("node_code", code)
            try:
                self.node_server = node.NodeServer(code=code, log=lambda m: self.bridge.log.emit("[node] " + m)).start()
            except OSError as e:
                self.node_mode_btn.setChecked(False)
                QMessageBox.warning(self, APP_NAME, f"Could not start the render node:\n{e}")
                return
            addr = f"{node.lan_ip()}:{node.DEFAULT_PORT}"
            self.node_mode_btn.setText(f"Render node: {addr} · code {code}")
            QMessageBox.information(
                self, APP_NAME,
                f"This PC is now a render node.\n\nOn the main PC, add a render node with:\n\n"
                f"    Address:        {addr}\n    Pairing code:  {code}\n\n"
                "If Windows Firewall asks, allow access on Private networks.\n"
                "Keep this window open while the other PC exports.")
        else:
            srv = getattr(self, "node_server", None)
            if srv:
                srv.stop()
                self.node_server = None
            self.node_mode_btn.setText("Render node: off")

    def _resume_info(self, it):
        """(done, total) segments of an unfinished full export of this item with the current settings."""
        if not self.engine:
            return None
        s = {"folder": self.folder_edit.text().strip(), "ext": self.container_box.currentText()}
        from upscaler import export
        return export.resumable_progress(self._output_path(it.path, s, "_4K"))

    def _export(self, selected_only):
        s = self._snapshot_settings()
        if not s["gpus"]:
            QMessageBox.warning(self, APP_NAME, "Select at least one GPU.")
            return
        if selected_only:
            chosen = [self.items[self.queue_list.row(li)] for li in self.queue_list.selectedItems()]
            chosen.sort(key=self.items.index)
        else:
            chosen = list(self.items)
        chosen = [i for i in chosen if (i.info or i.kind == "audio")
                  and i.status in ("Ready", "Done", "Failed", "Cancelled", "Stopped", "Skipped")]
        if not chosen:
            return
        from upscaler import audio
        needs_audio = any(i.kind == "audio" for i in chosen) or s["upscale_mode"] > 0
        if needs_audio and not audio.weights_present():
            QMessageBox.information(self, APP_NAME, "Restoring audio needs the AudioSR model (6.2 GB, downloaded "
                                    "once). Download it first, then export again.")
            self._download_audio_model()
            return
        self._discard_clip()            # free RAM and GPUs for the export
        wait = self.sched_check.isChecked() and not self._in_window() and not self._sched_override
        for it in chosen:
            it.settings = dict(s)
            if s["range"] and it.r_out > it.r_in >= 0:
                it.settings["start"], it.settings["seconds"] = it.r_in, it.r_out - it.r_in
            elif s["test"]:
                it.settings["start"], it.settings["seconds"] = 0.0, 10.0
            else:
                it.settings["start"], it.settings["seconds"] = 0.0, None
            it.progress = 0.0
            if wait:
                self._schedule_item(it)
            else:
                it.status, it.detail = "Queued", ""
                it.widget.refresh()
                self.export_q.put(it)
        if not wait and not self.exporting:
            self._start_export_thread()
        self._sched_refresh()
        self._update_buttons()

    # ================================================================ schedule (time window)
    def _in_window(self, now: QTime | None = None) -> bool:
        a, b = self.sched_start.time(), self.sched_end.time()
        now = now or QTime.currentTime()
        if a == b:
            return True                                  # same start and end = all day
        if a < b:
            return a <= now < b                          # e.g. 09:00-17:00
        return now >= a or now < b                       # overnight, e.g. 23:00-07:00

    def _secs_until_window(self) -> int:
        now = QTime.currentTime()
        d = now.secsTo(self.sched_start.time())
        return d if d >= 0 else d + 24 * 3600

    def _schedule_item(self, it, detail=None):
        it.status = "Scheduled"
        it.detail = detail or f"at {self.sched_start.time().toString('HH:mm')}"
        if it not in self.scheduled:
            self.scheduled.append(it)
        if it.widget:
            it.widget.refresh()
        self._queue_changed()

    def _sched_settings_changed(self, *_):
        st = self.settings
        st.setValue("sched_on", "true" if self.sched_check.isChecked() else "false")
        st.setValue("sched_start", self.sched_start.time().toString("HH:mm"))
        st.setValue("sched_end", self.sched_end.time().toString("HH:mm"))
        for it in self.scheduled:
            if not it.detail.startswith("paused"):
                it.detail = f"at {self.sched_start.time().toString('HH:mm')}"
                it.widget.refresh()
        self._sched_tick()

    def _sched_tick(self):
        """Every 10 s: start scheduled videos when the window opens, pause the queue when it closes."""
        if not hasattr(self, "scheduled"):
            return
        enabled = self.sched_check.isChecked()
        open_now = (not enabled) or self._in_window() or self._sched_override
        if open_now and self.scheduled and not self._sched_pausing:
            items, self.scheduled = self.scheduled, []
            for it in items:
                it.status, it.detail = "Queued", ""
                it.widget.refresh()
                self.export_q.put(it)
            self.log(f"Schedule: starting {len(items)} video(s).")
            if not self.exporting:
                self._start_export_thread()
        elif not open_now and self.exporting and not self._sched_pausing:
            self._sched_pausing = True
            self.log(f"Schedule: window ended at {self.sched_end.time().toString('HH:mm')} - pausing; "
                     "the queue continues in the next window.")
            while True:
                try:
                    it = self.export_q.get_nowait()
                except queue.Empty:
                    break
                self._schedule_item(it)
            if self.current_job:
                self.current_job.cancel()
        self._sched_refresh()
        self._update_buttons()

    def _sched_refresh(self):
        enabled = self.sched_check.isChecked()
        self.sched_start.setEnabled(enabled)
        self.sched_end.setEnabled(enabled)
        if not enabled:
            self.sched_hint.setText("Off - exports start right away.")
        else:
            a, b = self.sched_start.time().toString("HH:mm"), self.sched_end.time().toString("HH:mm")
            if self._in_window():
                state = f"Inside the window now (until {b})."
            else:
                secs = self._secs_until_window()
                state = f"Next window starts at {a} (in {secs // 3600} h {secs // 60 % 60:02d} min)."
            self.sched_hint.setText(f"{state} The PC is kept awake while videos are scheduled or exporting; "
                                    "keep this app open.")
        if self.scheduled and not self.exporting:
            secs = self._secs_until_window()
            self.status_text.setText(f"{len(self.scheduled)} video(s) scheduled - starting at "
                                     f"{self.sched_start.time().toString('HH:mm')} "
                                     f"(in {secs // 3600} h {secs // 60 % 60:02d} min)")
        self._keep_awake(self.exporting or bool(self.scheduled))

    def _start_scheduled_now(self):
        self._sched_override = True        # ignore the window until the queue is empty
        self._sched_tick()

    def _keep_awake(self, on: bool):
        """Ask Windows not to sleep (like a video player does). Changes no settings; released when idle."""
        if on == getattr(self, "_awake", False):
            return
        try:
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
            self._awake = on
            self.log("Keeping the PC awake for scheduled/running exports." if on else "PC may sleep again.")
        except Exception:  # noqa: BLE001
            pass

    def _start_export_thread(self):
        self.exporting = True
        self.export_thread = threading.Thread(target=self._export_loop, daemon=True)
        self.export_thread.start()
        self.status_bar.show()

    def _export_loop(self):
        while True:
            try:
                it = self.export_q.get_nowait()
            except queue.Empty:
                break
            if it.status != "Queued":
                continue
            from upscaler import audio_job, export
            s = it.settings
            if it.kind == "audio":
                self._export_audio_file(it, s)
                continue
            mode = s["upscale_mode"]
            if mode == 2:                                    # audio only: whole file, picture copied
                s = dict(s, start=0.0, seconds=None, range=False, test=False)
                it.output_path = self._output_path(it.path, s, "_audio")
            else:
                it.output_path = self._output_path(it.path, s, self._suffix(s))
            it.status, it.detail = "Processing", "starting"
            self.bridge.item_changed.emit(it)
            self.bridge.log.emit(f"=== {os.path.basename(it.path)} -> {it.output_path}")
            fps = float(it.info.fps)
            job = export.ExportJob(
                it.path, it.output_path, s["spec"], s["gpus"], s["encoder"], cq=s["cq"],
                start_frame=round(s["start"] * fps),
                frame_count=round(s["seconds"] * fps) if s["seconds"] else None,
                use_tensorrt=s["trt"], remote_workers=self._remote_workers(s) if mode != 2 else [],
                restore_audio=mode > 0, video=mode != 2, audio_steps=s["audio_steps"],
                on_log=lambda m, it=it: self._job_log(it, m),
                on_progress=lambda p, it=it: self.bridge.progress.emit(it, p),
            )
            self.current_job = job
            t0 = time.time()
            try:
                ok = job.run()
                if ok:
                    it.status, it.progress = "Done", 1.0
                    it.detail = fmt_time(time.time() - t0) + (" (resumed)" if job.resumed_segments else "")
                    if job.audio_note:
                        it.detail += f" · {job.audio_note}"
                else:
                    prog = export.resumable_progress(it.output_path)
                    pct = f"{prog[0] * 100 // max(prog[1], 1)}% saved" if prog else ""
                    if self._sched_pausing:
                        # paused by the schedule: continue first thing in the next window
                        it.status, it.detail = "Scheduled", f"paused, {pct}"
                        self.scheduled.insert(0, it)
                    else:
                        it.status = "Stopped"
                        it.detail = f"{pct} - export again to resume" if prog else ""
            except audio_job.NotNeeded as e:
                it.status, it.detail = "Skipped", f"audio {e}: nothing to restore"
                self.bridge.log.emit(f"Skipped: audio {e}.")
            except Exception as e:  # noqa: BLE001
                it.status, it.detail = "Failed", str(e).splitlines()[0][:120]
                prog = export.resumable_progress(it.output_path)
                if prog and prog[0]:
                    it.detail += f" ({prog[0]}/{prog[1]} segments saved - export again to resume)"
                self.bridge.log.emit(f"Failed: {e}")
            self.current_job = None
            self.bridge.item_changed.emit(it)
        self.bridge.export_idle.emit()

    def _export_audio_file(self, it, s):
        """Audio files: restore the treble and save in the chosen format (runs in the export thread)."""
        import threading as _th
        from upscaler import audio_job
        ext, codec = audio_job.AUDIO_FORMATS[s["audio_format"]]
        it.output_path = self._output_path(it.path, dict(s, ext=ext), "_restored")
        it.status, it.detail = "Processing", "analysing audio"
        self.bridge.item_changed.emit(it)
        self.bridge.log.emit(f"=== {os.path.basename(it.path)} -> {it.output_path}")
        cancel = _th.Event()

        class _AudioJobHandle:                           # lets Stop / the schedule cancel this job
            def cancel(self_):
                cancel.set()
        self.current_job = _AudioJobHandle()
        t0 = time.time()
        try:
            r = audio_job.restore_audio_file(
                it.path, it.output_path, s["gpus"], steps=s["audio_steps"], codec_args=codec, cancel_event=cancel,
                on_log=lambda m: self._job_log(it, m),
                on_progress=lambda d, t, eta: self.bridge.progress.emit(
                    it, self.engine.Progress(d, t, 0, eta, {"Audio (AudioSR)": d})))
            if r is False:
                it.status, it.detail = ("Scheduled", "paused") if self._sched_pausing else ("Stopped", "export again to resume")
                if self._sched_pausing:
                    self.scheduled.insert(0, it)
            else:
                it.status, it.progress = "Done", 1.0
                it.detail = f"{fmt_time(time.time() - t0)} · restored from {r / 1000:.1f} kHz"
        except audio_job.NotNeeded as e:
            it.status, it.detail = "Skipped", f"{e}: nothing to restore"
        except Exception as e:  # noqa: BLE001
            it.status, it.detail = "Failed", str(e).splitlines()[0][:120]
            self.bridge.log.emit(f"Failed: {e}")
        self.current_job = None
        self.bridge.item_changed.emit(it)

    def _job_log(self, it, msg):
        self.bridge.log.emit(msg)
        if msg.startswith(("Building TensorRT", "Exporting", "Downloading")):
            it.detail = msg.split(" - ")[0].rstrip(".")
            self.bridge.item_changed.emit(it)

    def _on_progress(self, it, p):
        if it.status != "Processing":      # a late update arriving after the job finished
            return
        if p.total:
            it.progress = min(p.done / p.total, 1.0)
        if "Audio (AudioSR)" in p.per_gpu:
            it.detail = f"restoring audio {it.progress * 100:.0f}%"
        else:
            it.detail = f"{it.progress * 100:.0f}%  ·  {p.fps:.1f} fps"
        it.widget.refresh()
        self.status_bar.setValue(int(it.progress * 1000))
        remaining = self.export_q.qsize()
        more = f"  ·  {remaining} more queued" if remaining else ""
        if "Audio (AudioSR)" in p.per_gpu:
            self.status_text.setText(f"Restoring audio of {os.path.basename(it.path)}  ·  chunk {p.done}/{p.total}  ·  "
                                     f"ETA {fmt_time(p.eta_s)}{more}")
            self.share_text.setText("AudioSR on all GPUs")
            return
        self.status_text.setText(f"Exporting {os.path.basename(it.path)}  ·  frame {p.done}/{p.total}  ·  "
                                 f"{p.fps:.2f} fps  ·  ETA {fmt_time(p.eta_s)}{more}")
        total = sum(p.per_gpu.values()) or 1
        self.share_text.setText("   ".join(f"{g if isinstance(g, str) else f'GPU {g}'}: {n * 100 / total:.0f}%"
                                           for g, n in sorted(p.per_gpu.items(), key=lambda kv: str(kv[0]))))

    def _export_idle(self):
        if not self.export_q.empty():   # videos were queued just as the last one finished
            self._start_export_thread()
            return
        self.exporting = False
        paused = self._sched_pausing
        self._sched_pausing = False
        self._sched_override = False
        self.status_bar.hide()
        self.share_text.setText("")
        done = [i for i in self.items if i.status == "Done"]
        self.status_text.setText(f"Export finished. {len(done)} video(s) done." if done else "Export stopped.")
        self.log("Export queue paused by the schedule." if paused else "Export queue finished.")
        if not self.isVisible():                     # running in the tray: tell the user
            failed = [i for i in self.items if i.status == "Failed"]
            if paused:
                self.notify("MUSS paused", f"The schedule window ended; {len(self.scheduled)} item(s) continue "
                                           f"at {self.sched_start.time().toString('HH:mm')}.")
            else:
                self.notify("MUSS queue finished", f"{len(done)} done" + (f", {len(failed)} failed" if failed else "") + ".")
        self._sched_refresh()
        self._update_buttons()
        self._queue_changed()

    def _stop(self):
        while True:
            try:
                it = self.export_q.get_nowait()
            except queue.Empty:
                break
            it.status, it.detail = "Ready", ""
            it.widget.refresh()
        for it in self.scheduled:                # Stop also cancels the schedule
            it.status, it.detail = "Ready", ""
            it.widget.refresh()
        self.scheduled = []
        self._sched_override = False
        if self.current_job:
            self.current_job.cancel()
            self.status_text.setText("Stopping...")
        self._sched_refresh()
        self._update_buttons()
        self._queue_changed()

    # ================================================================ tray + startup
    def _setup_tray(self):
        from PySide6.QtWidgets import QMenu, QSystemTrayIcon
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(make_icon(), self)
        self.tray.setToolTip(APP_TITLE)
        menu = QMenu()
        self.tray_status = menu.addAction("Idle")
        self.tray_status.setEnabled(False)
        menu.addSeparator()
        menu.addAction("Show MUSS", self._show_from_tray)
        self.tray_start_now = menu.addAction("Start scheduled videos now", self._start_scheduled_now)
        self.tray_stop = menu.addAction("Stop the queue", self._stop)
        menu.addSeparator()
        menu.addAction("Quit MUSS", self._quit_from_tray)
        menu.aboutToShow.connect(self._refresh_tray_menu)
        self.tray.setContextMenu(menu)
        self.tray_menu = menu
        self.tray.activated.connect(
            lambda reason: self._show_from_tray() if reason in (QSystemTrayIcon.ActivationReason.DoubleClick,
                                                                QSystemTrayIcon.ActivationReason.Trigger) else None)
        self.tray.show()

    def _refresh_tray_menu(self):
        if self.exporting:
            text = self.status_text.text()[:70] or "Exporting..."
        elif self.scheduled:
            text = f"{len(self.scheduled)} scheduled - starts at {self.sched_start.time().toString('HH:mm')}"
        else:
            text = "Idle"
        self.tray_status.setText(text)
        self.tray_start_now.setEnabled(bool(self.scheduled) and not self.exporting)
        self.tray_stop.setEnabled(self.exporting or bool(self.scheduled))
        if self.tray:
            self.tray.setToolTip(f"{APP_TITLE}\n{text}")

    def _show_from_tray(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _quit_from_tray(self):
        self._really_quit = True
        self._show_from_tray() if (self.exporting or self.scheduled) else None
        if not self.close():                       # the user cancelled the "stop and quit?" question
            self._really_quit = False
            return
        QApplication.quit()

    def notify(self, title, text):
        if self.tray:
            self.tray.showMessage(title, text, make_icon(), 8000)

    def _startup_toggled(self, on):
        try:
            set_startup(on)
            self.log("MUSS will start with Windows (in the tray)." if on else "MUSS removed from Windows startup.")
        except OSError as e:
            QMessageBox.warning(self, APP_NAME, f"Couldn't change the startup setting:\n{e}")
            self.startup_check.blockSignals(True)
            self.startup_check.setChecked(startup_enabled())
            self.startup_check.blockSignals(False)

    # ================================================================ queue persistence
    def _queue_changed(self, *_):
        if hasattr(self, "save_timer"):
            self.save_timer.start()

    def _save_queue(self):
        """Persist the queue (paths, status, settings, preview range) so it survives restarts and logins."""
        import json
        out = []
        for it in self.items:
            s = dict(it.settings)
            if s.get("spec") is not None and not isinstance(s.get("spec"), str):
                s["spec"] = s["spec"].key
            status = it.status
            if status in ("Processing", "Queued"):
                status = "Interrupted"             # was running/waiting when MUSS closed
            out.append({"path": it.path, "status": status, "detail": it.detail, "output": it.output_path,
                        "settings": s, "r_in": it.r_in, "r_out": it.r_out, "playhead": it.playhead})
        self.settings.setValue("queue", json.dumps(out))

    def _restore_queue(self):
        import json
        try:
            saved = json.loads(self.settings.value("queue", "[]"))
        except (TypeError, ValueError):
            saved = []
        missing = [e["path"] for e in saved if not os.path.isfile(e["path"])]
        for p in missing:
            self.log(f"Queue: {p} no longer exists - removed from the queue.")
        entries = [e for e in saved if os.path.isfile(e["path"])]
        if not entries:
            return
        self.add_files([e["path"] for e in entries])
        by_path = {os.path.normcase(os.path.normpath(e["path"])): e for e in entries}
        for it in self.items:
            it.restore = by_path.get(os.path.normcase(it.path))
        self.log(f"Restored {len(entries)} item(s) from the saved queue.")

    def _apply_restore(self, it):
        """Called once the item's file has been read: bring back its saved state."""
        e, it.restore = it.restore, None
        if not e or it.status != "Ready":
            return
        if e.get("r_out", -1) > e.get("r_in", -1) >= 0:
            it.r_in, it.r_out, it.playhead = e["r_in"], e["r_out"], e.get("playhead", e["r_in"])
        s = dict(e.get("settings") or {})
        if s.get("spec") is not None:
            spec = getattr(self, "specs", {}).get(s["spec"])
            if spec is None:
                it.status, it.detail = "Failed", f"model '{s['spec']}' is no longer installed"
                return
            s["spec"] = spec
        status = e.get("status")
        if status in ("Done", "Skipped", "Failed", "Stopped"):
            it.status, it.detail, it.output_path = status, e.get("detail", ""), e.get("output", "")
            it.progress = 1.0 if status == "Done" else 0.0
        elif status in ("Scheduled", "Interrupted") and s:
            it.settings = s
            if status == "Scheduled" or self.sched_check.isChecked():
                self._schedule_item(it, e.get("detail") if status == "Scheduled" else "interrupted - continues next window")
            else:
                it.status, it.detail = "Stopped", "interrupted - export again to resume"
        self._sched_refresh()
        self._sched_tick()

    # ================================================================ log + close
    def log(self, msg):
        self._append_log(msg)

    def _append_log(self, msg):
        self.log_view.appendPlainText(time.strftime("%H:%M:%S  ") + msg)
        if msg.startswith(("Error", "Failed", "AI preview failed", "Could not", "Preview failed")):
            self.status_text.setText(msg.splitlines()[0][:160])

    def closeEvent(self, e):
        self._save_queue()
        if not self._really_quit and self.tray and self.tray_check.isChecked():
            e.ignore()                              # keep running in the tray
            self.hide()
            if not self._tray_hint_shown:
                self._tray_hint_shown = True
                busy = "Exports and the schedule keep running. " if (self.exporting or self.scheduled) else ""
                self.notify("MUSS is still running", f"{busy}Open or quit MUSS from the tray icon.")
            return
        if self.scheduled and not self.exporting:
            if QMessageBox.question(self, APP_NAME, f"{len(self.scheduled)} video(s) are scheduled. They only run "
                                    "while MUSS is running (it can stay in the tray).\n\nQuit anyway?") \
                    != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
        self._keep_awake(False)
        if self.exporting:
            if QMessageBox.question(self, APP_NAME, "An export is running. Stop it and quit?") != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self._stop()
            if self.export_thread:
                self.export_thread.join(timeout=10)
        if getattr(self, "node_server", None):
            self.node_server.stop()
        self._discard_clip()
        self._save_queue()
        if self.tray:
            self.tray.hide()
        e.accept()
        QApplication.quit()


def _show_splash(app):
    """The MUSS banner while the AI engine loads (a few seconds)."""
    from PySide6.QtWidgets import QSplashScreen
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "banner.jpg")
    pm = QPixmap(path)
    if pm.isNull():
        return None
    screen = app.primaryScreen()
    dpr = screen.devicePixelRatio() if screen else 1.0
    width = 800
    if screen:
        width = min(width, int(screen.availableGeometry().width() * 0.6))
    pm = pm.scaledToWidth(int(width * dpr), Qt.TransformationMode.SmoothTransformation)
    pm.setDevicePixelRatio(dpr)
    splash = QSplashScreen(pm, Qt.WindowType.WindowStaysOnTopHint)
    f = QFont("Segoe UI", 9)
    splash.setFont(f)
    splash.show()
    splash.showMessage("Loading the AI engine...  ", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
                       QColor("#c8cfdc"))
    app.processEvents()
    return splash


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setQuitOnLastWindowClosed(False)            # MUSS can keep running in the tray
    app.setWindowIcon(make_icon())
    tray_start = "--tray" in sys.argv[1:]
    splash = None if tray_start else _show_splash(app)
    w = MainWindow()
    if splash:
        def close_splash(*_):
            if splash.isVisible():
                splash.finish(w)
        w.bridge.engine_ready.connect(close_splash)  # connected after the window's own handler: runs after it
        w.bridge.engine_failed.connect(close_splash)
        QTimer.singleShot(30000, close_splash)       # never leave it up if something goes wrong
    if tray_start and w.tray:                        # started with Windows: stay hidden in the tray
        w._tray_hint_shown = True
    else:
        w.show()
        if splash:
            splash.raise_()
    w.add_files([a for a in sys.argv[1:] if os.path.isfile(a)])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
