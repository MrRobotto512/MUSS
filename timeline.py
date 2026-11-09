"""Timeline widget: filmstrip, time ruler, preview range (in/out handles), rendered bar, playhead."""
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

ACCENT = QColor("#4f8cff")
RANGE = QColor("#f5b74f")
RENDERED = QColor("#3ecf8e")


def fmt_tc(t: float, fps: float) -> str:
    """Timecode HH:MM:SS:FF."""
    fps = max(fps, 1.0)
    total = int(round(max(t, 0) * fps))
    f = total % round(fps)
    s = int(total // round(fps))
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{f:02d}"


class RangeTimeline(QWidget):
    playheadMoved = Signal(float, bool)     # seconds, final (mouse released)
    rangeChanged = Signal(float, float)     # in, out seconds

    RULER = 18
    STRIP = 52
    BAR = 6
    HANDLE = 9

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(self.RULER + self.STRIP + self.BAR + 10)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.duration = 0.0
        self.fps = 30.0
        self.playhead = 0.0
        self.r_in, self.r_out = 0.0, 0.0
        self.thumbs: list[tuple[float, QPixmap]] = []
        self.rendered = None          # (start, end) seconds
        self.rendering = None         # (start, end) seconds being rendered
        self._drag = None

    # ------------------------------------------------------------------ API
    def set_media(self, duration, fps):
        self.duration, self.fps = max(duration, 0.0), max(fps, 1.0)
        self.thumbs = []
        self.rendered = self.rendering = None
        self.update()

    def set_thumbs(self, thumbs):
        self.thumbs = sorted(thumbs, key=lambda x: x[0])
        self.update()

    def set_playhead(self, t):
        self.playhead = min(max(t, 0.0), self.duration)
        self.update()

    def set_range(self, a, b):
        self.r_in, self.r_out = self._clamp_range(a, b)
        self.update()

    def set_rendered(self, span):
        self.rendered = span
        self.update()

    def set_rendering(self, span):
        self.rendering = span
        self.update()

    # ------------------------------------------------------------------ geometry
    def _lane(self):
        return QRectF(8, 0, max(self.width() - 16, 1), self.height())

    def _x(self, t):
        lane = self._lane()
        return lane.left() + (t / self.duration * lane.width() if self.duration else 0)

    def _t(self, x):
        lane = self._lane()
        if not self.duration:
            return 0.0
        return min(max((x - lane.left()) / lane.width() * self.duration, 0.0), self.duration)

    def _clamp_range(self, a, b):
        min_len = 1.0 / self.fps
        a = min(max(a, 0.0), self.duration)
        b = min(max(b, a + min_len), self.duration)
        if b - a < min_len:
            a = max(b - min_len, 0.0)
        return a, b

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        lane = self._lane()
        strip = QRectF(lane.left(), self.RULER, lane.width(), self.STRIP)
        p.fillRect(self.rect(), QColor("#121419"))
        p.fillRect(strip, QColor("#0b0d11"))
        if not self.duration:
            return

        # filmstrip
        if self.thumbs:
            p.save()
            p.setClipRect(strip)
            for i, (t, pix) in enumerate(self.thumbs):
                x0 = self._x(t)
                x1 = self._x(self.thumbs[i + 1][0]) if i + 1 < len(self.thumbs) else strip.right()
                target = QRectF(x0, strip.top(), max(x1 - x0, 1), strip.height())
                src = QRectF(pix.rect())
                # crop to fill the cell without stretching
                ar_t, ar_s = target.width() / target.height(), src.width() / src.height()
                if ar_s > ar_t:
                    cw = src.height() * ar_t
                    src = QRectF(src.left() + (src.width() - cw) / 2, src.top(), cw, src.height())
                else:
                    ch = src.width() / ar_t
                    src = QRectF(src.left(), src.top() + (src.height() - ch) / 2, src.width(), ch)
                p.drawPixmap(target, pix, src)
            p.restore()

        # ruler
        f = QFont(self.font())
        f.setPointSize(8)
        p.setFont(f)
        step = self._tick_step(lane.width())
        p.setPen(QColor("#5d6575"))
        t = 0.0
        while t <= self.duration + 1e-6:
            x = self._x(t)
            p.drawLine(QPointF(x, self.RULER - 5), QPointF(x, self.RULER - 1))
            label = self._short(t)
            if x + 40 < lane.right() + 8:
                p.drawText(QPointF(x + 3, self.RULER - 6), label)
            t += step

        # dim outside the preview range, outline the range
        xi, xo = self._x(self.r_in), self._x(self.r_out)
        dim = QColor(0, 0, 0, 150)
        p.fillRect(QRectF(strip.left(), strip.top(), xi - strip.left(), strip.height()), dim)
        p.fillRect(QRectF(xo, strip.top(), strip.right() - xo, strip.height()), dim)
        p.setPen(QPen(RANGE, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(xi, strip.top() + 1, xo - xi, strip.height() - 2))
        for x, side in ((xi, -1), (xo, 1)):
            h = QRectF(x - self.HANDLE / 2, strip.top(), self.HANDLE, strip.height())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(RANGE)
            p.drawRoundedRect(h, 3, 3)
            p.setPen(QPen(QColor("#5a3f10"), 1))
            for dy in (-6, 0, 6):
                cy = strip.center().y() + dy
                p.drawLine(QPointF(x - 1.5, cy), QPointF(x + 1.5, cy))

        # rendered / rendering bars
        bar_y = strip.bottom() + 3
        p.fillRect(QRectF(strip.left(), bar_y, strip.width(), self.BAR), QColor("#1e222a"))
        if self.rendering:
            a, b = self.rendering
            p.fillRect(QRectF(self._x(a), bar_y, self._x(b) - self._x(a), self.BAR), QColor(RENDERED.red(), RENDERED.green(), RENDERED.blue(), 70))
        if self.rendered:
            a, b = self.rendered
            p.fillRect(QRectF(self._x(a), bar_y, max(self._x(b) - self._x(a), 2), self.BAR), RENDERED)

        # playhead
        x = self._x(self.playhead)
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(QPointF(x, 2), QPointF(x, bar_y + self.BAR))
        tri = QPainterPath()
        tri.moveTo(x - 6, 0)
        tri.lineTo(x + 6, 0)
        tri.lineTo(x, 8)
        tri.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#ffffff"))
        p.drawPath(tri)

    def _tick_step(self, width):
        for step in (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
            if self.duration / step * 60 <= width:
                return step
        return 3600

    @staticmethod
    def _short(t):
        s = int(t)
        return f"{s // 60}:{s % 60:02d}" if s < 3600 else f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"

    # ------------------------------------------------------------------ interaction
    def _hit(self, pos):
        strip_top, strip_bot = self.RULER, self.RULER + self.STRIP
        if strip_top <= pos.y() <= strip_bot:
            for name, t in (("in", self.r_in), ("out", self.r_out)):
                if abs(pos.x() - self._x(t)) <= self.HANDLE:
                    return name
        return "playhead"

    def mousePressEvent(self, e):
        if not self.duration or e.button() != Qt.MouseButton.LeftButton:
            return
        self._drag = self._hit(e.position())
        self._apply(e.position(), final=False)

    def mouseMoveEvent(self, e):
        if self._drag:
            self._apply(e.position(), final=False)
        else:
            hit = self._hit(e.position()) if self.duration else None
            self.setCursor(Qt.CursorShape.SizeHorCursor if hit in ("in", "out") else Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, e):
        if self._drag:
            self._apply(e.position(), final=True)
        self._drag = None

    def _apply(self, pos, final):
        t = self._t(pos.x())
        t = round(t * self.fps) / self.fps        # snap to frames
        if self._drag == "in":
            self.set_range(min(t, self.r_out - 1 / self.fps), self.r_out)
            self.rangeChanged.emit(self.r_in, self.r_out)
            self.set_playhead(self.r_in)
            self.playheadMoved.emit(self.playhead, final)
        elif self._drag == "out":
            self.set_range(self.r_in, max(t, self.r_in + 1 / self.fps))
            self.rangeChanged.emit(self.r_in, self.r_out)
            self.set_playhead(max(self.r_out - 1 / self.fps, self.r_in))
            self.playheadMoved.emit(self.playhead, final)
        else:
            self.set_playhead(t)
            self.playheadMoved.emit(self.playhead, final)
