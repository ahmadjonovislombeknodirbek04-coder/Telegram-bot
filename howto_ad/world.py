"""The one universe: layout, camera move, hero planes, the red line, and thousands of videos.

World units: x right, y down, z forward. With the 1500 px focal length a plane held
1450 units from the lens is drawn at ~1.03 px per unit on the 1920 x 1080 frame.
"""
import math

import numpy as np

from . import assets as A
from .core import (CameraRig, Pchip, clamp01, ease_out_back, ease_out_cubic, ease_out_expo, lin,
                   plane_corners, smooth, smoother, window)

FPS = 60
MAIN_DUR = 17.9
END_DUR = 1.7
DURATION = MAIN_DUR + END_DUR
FOCAL = 1500.0
WIDTH, HEIGHT = 1920, 1080

# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
ICON_C, ICON_W, ICON_H = np.array([0.0, -30.0, 0.0]), 380.0, 266.0
LINE0_Y = 143.0

SEARCH_C, SEARCH_W, SEARCH_H = np.array([0.0, -190.0, 1700.0]), 1180.0, 104.0
INPUT_X0, INPUT_X1 = -SEARCH_W / 2, SEARCH_W / 2 - 150
FOCUS_Y = SEARCH_C[1] + SEARCH_H / 2 + 1
TYPED = "how to"
TYPE_T = [2.02, 2.12, 2.21, 2.33, 2.41, 2.52]
SUGGEST = ["tie a tie", "make sourdough bread", "fix a bike chain", "start a youtube channel", "draw anything"]
PANEL_TOP = FOCUS_Y + 13
ROW_H = 64

PW, PH = 960.0, 540.0
V1_Z, P_Z, M_Z, LAST_Z = 5600.0, 13500.0, 17200.0, 20800.0
PLAYER_X, PLAYER_Y = 280.0, -60.0
BAR_X0 = PLAYER_X - PW / 2 + 20
BAR_X1 = PLAYER_X + PW / 2 - 20
BAR_Y = PLAYER_Y + PH / 2 - 54

CHAPTERS = [0.0, 0.14, 0.38, 0.63, 0.84, 1.0]
CHAPTER_NAMES = ["Intro", "Cut the legs", "Join the frame", "Sand it smooth", "Finish"]
JUMPS = [(8.62, 1), (9.12, 2), (9.62, 3), (10.1, 4)]  # (time, chapter index jumped to)

VF_X = -280.0
RING_C = np.array([VF_X, PLAYER_Y + PH / 2 - 70, M_Z - 1])
RING_R = 46.0
UP_Y = PLAYER_Y + PH / 2 + 26
UP_X0, UP_X1 = VF_X - PW / 2, VF_X + PW / 2
REC_T0, REC_T1 = 11.05, 12.12
UP_T0, UP_T1 = 12.55, 13.4

LAST_TEXT, LAST_SIZE = "How to Anything.", 150.0
LAST_Y = 40.0
LOGO_Y = -205.0
LOGO_W, LOGO_H = 196.0, 138.0
WORDMARK_SIZE = 124.0

SUPER_SIZE = 124.0

# ----------------------------------------------------------------------------
# Camera: one continuous move. z rises through every key, PCHIP never overshoots,
# so the camera never stops and never backs up.
# ----------------------------------------------------------------------------
CAM_KEYS = [
    # t      position                  look target              roll
    (0.00, (-170.0, -20.0, -720.0), (20.0, 10.0, 0.0), -5.0),
    (0.70, (-70.0, 20.0, -470.0), (0.0, 40.0, 500.0), -2.0),
    (1.35, (0.0, 40.0, -60.0), (0.0, -40.0, 1700.0), 0.0),
    (2.10, (0.0, -10.0, 240.0), (0.0, -60.0, 1700.0), 0.0),
    (3.95, (0.0, -15.0, 400.0), (0.0, -50.0, 1700.0), 0.0),
    (4.60, (40.0, -10.0, 2900.0), (60.0, -20.0, V1_Z), 2.0),
    (5.15, (30.0, 0.0, 4100.0), (20.0, -10.0, V1_Z), 0.0),
    (6.05, (20.0, 0.0, 4290.0), (20.0, -10.0, V1_Z), 0.0),
    (6.90, (-450.0, 150.0, 8600.0), (-300.0, 60.0, 11500.0), 10.0),
    (7.80, (40.0, 10.0, 11900.0), (20.0, -10.0, P_Z), -2.0),
    (10.30, (20.0, 0.0, 12110.0), (20.0, -10.0, P_Z), 0.0),
    (11.05, (-30.0, 0.0, 15620.0), (-20.0, -10.0, M_Z), 0.0),
    (13.45, (-20.0, 0.0, 15800.0), (-20.0, -10.0, M_Z), 0.0),
    (14.15, (0.0, 20.0, 19050.0), (0.0, 30.0, LAST_Z), 0.0),
    (15.90, (0.0, 12.0, 19200.0), (0.0, 22.0, LAST_Z), 0.0),
    (17.90, (0.0, -60.0, 19330.0), (0.0, -60.0, LAST_Z), 0.0),
    (19.60, (0.0, -62.0, 19420.0), (0.0, -62.0, LAST_Z), 0.0),
]

# ----------------------------------------------------------------------------
# Hero planes
# ----------------------------------------------------------------------------


class Hero:
    def __init__(self, name, center, size, ppu, key, paint, alpha, offset=None, rot=(0, 0, 0), sharp=False):
        self.name = name
        self.center = np.asarray(center, float)
        self.size = size
        self.ppu = ppu
        self.key = key          # t -> hashable texture key
        self.paint = paint      # key -> premultiplied RGBA texture
        self.alpha = alpha      # t -> opacity
        self.offset = offset    # t -> world offset
        self.rot = rot
        self.sharp = sharp      # ignore depth-of-field (held type)

    def corners(self, t):
        c = self.center + (self.offset(t) if self.offset else 0)
        return plane_corners(c, self.size[0], self.size[1], self.rot)[0]


def _const(v):
    return lambda t: v


def _slide_up(t0, dur=0.5, dist=22.0):
    return lambda t: np.array([0.0, dist * (1 - ease_out_cubic(lin(t, t0, t0 + dur))), 0.0])


def _mmss(s):
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def v1_progress(t):
    return 0.06 + 0.52 * smooth(lin(t, 4.62, 6.0))


def p_progress(t):
    p = 0.035 + 0.06 * lin(t, 7.95, 8.62)
    for tj, idx in JUMPS:
        k = ease_out_expo(lin(t, tj, tj + 0.16))
        p = p + (CHAPTERS[idx] + 0.012 - p) * k
    return p + 0.03 * lin(t, 10.1, 10.9)


def p_chapter(t):
    idx = 0
    for tj, i in JUMPS:
        if t >= tj + 0.04:
            idx = i
    return idx


def upload_progress(t):
    return smoother(lin(t, UP_T0, UP_T1))


def logo_layout():
    ww = A.text_size("YouTube", WORDMARK_SIZE, "bold", True)[0]
    gap = 30.0
    total = LOGO_W + gap + ww
    left = -total / 2
    icon_x = left + LOGO_W / 2
    word_x = left + LOGO_W + gap + ww / 2
    return icon_x, word_x


def build_heroes():
    H = []
    # --- the play icon (opening) -----------------------------------------
    H.append(Hero("icon", ICON_C, (ICON_W, ICON_H), 2.6, _const("icon"),
                  lambda k: A.icon_texture(ICON_W, ICON_H, 2.6, base_alpha=0.0),
                  lambda t: 1 - smooth(lin(t, 1.0, 1.35))))

    # --- search ----------------------------------------------------------
    def search_key(t):
        n = sum(1 for tt in TYPE_T if t >= tt)
        typing = TYPE_T[0] - 0.2 <= t <= TYPE_T[-1] + 0.25
        caret = typing or ((t - TYPE_T[-1]) % 1.0) < 0.55
        return (n, bool(caret))

    H.append(Hero("search", SEARCH_C, (SEARCH_W, SEARCH_H), 1.3, search_key,
                  lambda k: A.search_texture(SEARCH_W, SEARCH_H, 1.3, TYPED[:k[0]], k[1]),
                  lambda t: window(t, 1.75, 4.05, 0.5, 0.3), _slide_up(1.25, 0.8, 40)))
    panel_h = ROW_H * len(SUGGEST) + 20
    pc = np.array([(INPUT_X0 + INPUT_X1) / 2, PANEL_TOP + panel_h / 2, SEARCH_C[2] + 2])
    pw = INPUT_X1 - INPUT_X0
    H.append(Hero("panel", pc, (pw, panel_h), 1.3, _const("panel"),
                  lambda k: A.panel_texture(pw, panel_h, 1.3),
                  lambda t: window(t, 2.78, 4.0, 0.22, 0.3), _slide_up(2.56, 0.4, 14)))
    for i, comp in enumerate(SUGGEST):
        t0 = 2.86 + 0.075 * i
        rc = np.array([pc[0], PANEL_TOP + 10 + ROW_H / 2 + ROW_H * i, SEARCH_C[2] + 1])
        H.append(Hero(f"row{i}", rc, (pw, ROW_H), 1.3, _const(comp),
                      (lambda c: (lambda k: A.suggestion_texture(pw, ROW_H, 1.3, "how to ", c, False)))(comp),
                      (lambda a: (lambda t: window(t, a + 0.2, 4.0, 0.2, 0.3)))(t0), _slide_up(t0, 0.4, 16)))

    # --- V1: watching ------------------------------------------------------
    pasta = A.frame_art("pasta", 1248, 702)
    v1_alpha = lambda t: window(t, 4.55, 6.15, 0.45, 0.45)
    H.append(Hero("v1", (PLAYER_X, PLAYER_Y, V1_Z), (PW, PH), 1.3,
                  lambda t: _mmss(v1_progress(t) * 698),
                  lambda k: A.player_texture(PW, PH, 1.3, pasta, f"{k} / 11:38"), v1_alpha))
    H.append(Hero("v1info", (PLAYER_X, PLAYER_Y + PH / 2 + 22 + 75, V1_Z), (PW, 150), 1.3, _const("i"),
                  lambda k: A.watch_info_texture(PW, 150, 1.3, "How to Make Fresh Pasta (No Machine)",
                                                 "The Patient Cook", "1.2M subscribers"),
                  v1_alpha, _slide_up(4.7, 0.6, 18)))
    H.extend(_super(["Watch it."], V1_Z, "right", 4.82, 6.05))

    # --- P: learning ------------------------------------------------------
    stool = A.frame_art("stool", 1248, 702)
    p_alpha = lambda t: window(t, 7.75, 10.45, 0.4, 0.4)
    H.append(Hero("p", (PLAYER_X, PLAYER_Y, P_Z), (PW, PH), 1.3,
                  lambda t: (_mmss(p_progress(t) * 845), p_chapter(t)),
                  lambda k: A.player_texture(PW, PH, 1.3, stool, f"{k[0]} / 14:05", CHAPTERS,
                                             CHAPTER_NAMES[k[1]]), p_alpha))
    H.append(Hero("pinfo", (PLAYER_X, PLAYER_Y + PH / 2 + 22 + 75, P_Z), (PW, 150), 1.3, p_chapter,
                  lambda k: A.chapter_info_texture(PW, 150, 1.3, "How to Build a Wooden Stool",
                                                   k + 1, len(CHAPTER_NAMES), CHAPTER_NAMES[k]),
                  p_alpha, _slide_up(7.9, 0.6, 18)))
    H.extend(_super(["Learn it."], P_Z, "right", 8.05, 10.35))

    # --- M: making ---------------------------------------------------------
    selfie = A.frame_art("selfie", 1248, 702)
    m_alpha = lambda t: window(t, 10.85, 13.75, 0.45, 0.4)

    def vf_key(t):
        rec = REC_T0 <= t < REC_T1
        secs = 3 + int((t - REC_T0) * 6) if rec else 0
        dot = rec and ((t - REC_T0) % 0.5) < 0.32
        disk = round(float(smooth(lin(t, REC_T0 - 0.12, REC_T0 + 0.1))
                           * (1 - smooth(lin(t, REC_T1, REC_T1 + 0.2)))), 2)
        square = round(float(ease_out_back(lin(t, REC_T0 + 0.05, REC_T0 + 0.3)) if rec else 0.0), 2)
        status = "Published" if t >= UP_T1 + 0.05 else ("Uploading" if t >= UP_T0 - 0.2 else "")
        return (rec, f"00:00:{secs:02d}", dot, disk, square, status)

    H.append(Hero("vf", (VF_X, PLAYER_Y, M_Z), (PW, PH), 1.3, vf_key,
                  lambda k: A.viewfinder_texture(PW, PH, 1.3, selfie, *k), m_alpha))
    H.append(Hero("uptrack", (VF_X, UP_Y, M_Z + 1), (PW, 6), 1.3, _const("tr"),
                  lambda k: A.track_texture(PW, 6, 1.3),
                  lambda t: m_alpha(t) * smooth(lin(t, 12.25, 12.55))))

    def up_key(t):
        if t < UP_T0 - 0.25:
            return ("", "", "")
        p = upload_progress(t)
        if t >= UP_T1 + 0.05:
            return ("How to Fix Anything", "Your first video", "Published")
        return ("How to Fix Anything", "Your first video", f"Uploading {int(p * 100)}%")

    H.append(Hero("upinfo", (VF_X, UP_Y + 66, M_Z), (PW, 110), 1.3, up_key,
                  lambda k: A.upload_info_texture(PW, 110, 1.3, *k),
                  lambda t: m_alpha(t) * smooth(lin(t, 12.25, 12.6)), _slide_up(12.25, 0.5, 14)))
    H.extend(_super(["Make it", "yours."], M_Z, "left", 11.15, 13.6))

    # --- last line + logo ----------------------------------------------------
    tex, (tw, th) = A.super_texture([LAST_TEXT], LAST_SIZE, 1.15, "center", weight="bold")
    H.append(Hero("last", (0, LAST_Y, LAST_Z), (tw, th), 1.15, _const("last"), lambda k, tex=tex: tex,
                  lambda t: smooth(lin(t, 13.75, 14.3)), _slide_up(13.75, 0.7, 26), sharp=True))
    icon_x, word_x = logo_layout()

    def logo_key(t):
        buf = round(float(smoother(lin(t, 16.42, 16.98))), 3)
        fill = round(float(smoother(lin(t, 16.58, 17.22))), 3)
        tri = round(float(smooth(lin(t, 17.18, 17.32))), 2)
        sc = round(float(ease_out_back(lin(t, 17.18, 17.5), 2.2)), 3)
        return (buf, fill, tri, sc)

    H.append(Hero("logo", (icon_x, LOGO_Y, LAST_Z - 1), (LOGO_W, LOGO_H), 2.0, logo_key,
                  lambda k: A.icon_texture(LOGO_W, LOGO_H, 2.0, fill=k[1], buffered=k[0], tri=k[2],
                                           tri_scale=0.6 + 0.4 * k[3], base_alpha=1.0),
                  lambda t: smooth(lin(t, 16.2, 16.5)), sharp=True))
    wtex, (ww, wh) = A.super_texture(["YouTube"], WORDMARK_SIZE, 1.3, "center", A.WHITE, "bold")
    word_y = LOGO_Y + (wh / 2 - (0.95 - 0.365) * WORDMARK_SIZE)
    H.append(Hero("wordmark", (word_x, word_y, LAST_Z - 1), (ww, wh), 1.3, _const("w"), lambda k: wtex,
                  lambda t: smooth(lin(t, 17.32, 17.8)),
                  lambda t: np.array([-36 * (1 - ease_out_cubic(lin(t, 17.32, 17.9))), 0, 0]), sharp=True))
    return H


def _super(lines, z, align, t0, t1):
    tex, (w, h) = A.super_texture(lines, SUPER_SIZE, 1.3, align)
    gap = 70.0
    if align == "right":   # sits left of a player on the right
        cx = PLAYER_X - PW / 2 - gap - w / 2
    else:                  # sits right of the viewfinder on the left
        cx = VF_X + PW / 2 + gap + w / 2
    cy = PLAYER_Y - 30
    return [Hero(f"super{z}", (cx, cy, z), (w, h), 1.3, _const("s"), lambda k: tex,
                 lambda t: window(t, t0 + 0.35, t1, 0.35, 0.3), _slide_up(t0, 0.75, 30), sharp=True)]


def last_line_metrics():
    tw, asc, desc = A.text_size(LAST_TEXT, LAST_SIZE, "bold", True)
    _, (pw, ph) = A.super_texture([LAST_TEXT], LAST_SIZE, 1.0, "center", weight="bold")
    baseline = LAST_Y - ph / 2 + LAST_SIZE * 0.95
    return tw, baseline


# ----------------------------------------------------------------------------
# The red line: one continuous line of light that does every job.
# Each state is a resampled world-space polyline; states hand over by morphing.
# ----------------------------------------------------------------------------
NL = 160


def _seg(a, b, f0, f1):
    u = np.linspace(f0, f1, NL)[:, None]
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + (b - a) * u


def _rrect_path(cx, cy, z, w, h, r, frac, n=NL):
    """Perimeter of a rounded rect, starting bottom-centre, running clockwise (on screen)."""
    pts = []
    m = 24
    pts.append((cx, cy + h / 2))
    # bottom-right corner (angle 90 -> 0), right edge, top-right (0 -> -90), top, top-left, left, bottom-left
    arcs = [((cx + w / 2 - r, cy + h / 2 - r), 90, 0), ((cx + w / 2 - r, cy - h / 2 + r), 0, -90),
            ((cx - w / 2 + r, cy - h / 2 + r), -90, -180), ((cx - w / 2 + r, cy + h / 2 - r), -180, -270)]
    for (ax, ay), a0, a1 in arcs:
        for k in range(m + 1):
            a = math.radians(a0 + (a1 - a0) * k / m)
            pts.append((ax + r * math.cos(a), ay + r * math.sin(a)))
    pts.append((cx, cy + h / 2))
    pts = np.array(pts)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.linspace(0, d[-1] * frac, n)
    x = np.interp(s, d, pts[:, 0])
    y = np.interp(s, d, pts[:, 1])
    return np.stack([x, y, np.full(n, z)], 1)


class Line:
    def __init__(self, rig):
        self.rig = rig
        tw, baseline = last_line_metrics()
        self.under_y = baseline + LAST_SIZE * 0.2
        self.under_x = tw / 2
        self.icon_x = logo_layout()[0]
        self.p0 = np.array([BAR_X0 + (BAR_X1 - BAR_X0) * v1_progress(6.0), BAR_Y, V1_Z - 1])
        self.p1 = np.array([BAR_X0 + (BAR_X1 - BAR_X0) * p_progress(7.95), BAR_Y, P_Z - 1])
        # comet head samples for card ignition
        ts = np.linspace(6.0, 7.75, 400)
        self.head_t = ts
        self.head_p = np.array([self.head(t) for t in ts])
        self.states = [  # (state fn, active from, active until) – gaps between are morphs
            (self.s_birth, 0.0, 0.98),
            (self.s_focus, 1.98, 4.0),
            (self.s_v1, 4.62, 6.0),
            (self.s_comet, 6.36, 7.52),
            (self.s_p, 7.98, 10.32),
            (self.s_ring, 10.96, 12.12),
            (self.s_upload, 12.5, 13.5),
            (self.s_under, 14.12, 15.9),
            (self.s_logo, 16.45, 99.0),
        ]

    # -- comet head: rides ahead of the camera on a slow helix -------------
    def head_base(self, t):
        cam = self.rig.at(t + 0.5)
        th = 2.2 + 2 * math.pi * 0.62 * (t - 6.0)
        off = np.array([380 * math.cos(th), 250 * math.sin(th) + 40, 0.0])
        return cam.pos + cam.R.T @ off

    def head(self, t):
        t = min(max(t, 6.0), 7.75)
        hb = self.head_base(t)
        a = smoother(lin(t, 6.0, 6.5))
        b = smoother(lin(t, 7.2, 7.68))
        p = self.p0 + (hb - self.p0) * a
        return p + (self.p1 - p) * b

    # -- states: return (points, weights, world width, knob) ---------------
    def s_birth(self, t):
        hw = 540 * ease_out_cubic(lin(t, 0.02, 0.98)) + 3
        z = ICON_C[2] - 1
        return _seg((-hw, LINE0_Y, z), (hw, LINE0_Y, z), 0, 1), np.ones(NL), 6.0, 0.0

    def s_focus(self, t):
        z = SEARCH_C[2] - 3
        return _seg((INPUT_X0 + 30, FOCUS_Y, z), (INPUT_X1 - 2, FOCUS_Y, z), 0, 1), np.ones(NL), 4.5, 0.0

    def s_v1(self, t):
        z = V1_Z - 1
        return _seg((BAR_X0, BAR_Y, z), (BAR_X1, BAR_Y, z), 0, v1_progress(t)), np.ones(NL), 5.5, 1.0

    def s_comet(self, t):
        taus = t - 0.34 * (1 - np.linspace(0, 1, NL)) ** 1.2
        pts = np.array([self.head(tau) for tau in taus])
        w = np.linspace(0, 1, NL) ** 1.4
        return pts, w, 9.0, 1.0

    def s_p(self, t):
        z = P_Z - 1
        return _seg((BAR_X0, BAR_Y, z), (BAR_X1, BAR_Y, z), 0, p_progress(t)), np.ones(NL), 5.5, 1.0

    def s_ring(self, t):
        a = math.pi / 2 + np.linspace(0, 2 * math.pi, NL)
        pts = np.stack([RING_C[0] + RING_R * np.cos(a), RING_C[1] + RING_R * np.sin(a), np.full(NL, RING_C[2])], 1)
        return pts, np.ones(NL), 7.0, 0.0

    def s_upload(self, t):
        z = M_Z - 1
        p = 0.004 + upload_progress(t) * 0.996
        return _seg((UP_X0, UP_Y, z), (UP_X1, UP_Y, z), 0, p), np.ones(NL), 6.0, 0.6

    def s_under(self, t):
        z = LAST_Z - 1
        return (_seg((-self.under_x, self.under_y, z), (self.under_x, self.under_y, z), 0, 1),
                np.ones(NL), 8.0, 0.0)

    def s_logo(self, t):
        pts = _rrect_path(self.icon_x, LOGO_Y, LAST_Z - 2, LOGO_W, LOGO_H, LOGO_H * 0.235, 1.0)
        return pts, np.ones(NL), 6.0, 0.0

    def gain(self, t):
        g = 1.0
        g *= smooth(lin(t, 0.0, 0.05))
        g *= 1 - smooth(lin(t, 17.12, 17.42))  # merges into the filled logo
        g += 1.6 * math.exp(-max(0.0, t - 0.06) / 0.18) * (t >= 0.0)  # birth flash
        for tj, _ in JUMPS:
            if t >= tj:
                g += 0.9 * math.exp(-(t - tj) / 0.12)
        if REC_T0 <= t:
            g += 0.6 * math.exp(-(t - REC_T0) / 0.2)
        return g

    def at(self, t):
        """(points (N,3), weights (N,), width, knob, gain)."""
        st = self.states
        for i, (fn, a, b) in enumerate(st):
            if a <= t <= b:
                p, w, wd, kn = fn(t)
                return p, w, wd, kn, self.gain(t)
            if i + 1 < len(st) and b < t < st[i + 1][1]:
                nfn, na, _ = st[i + 1]
                u = (t - b) / (na - b)
                p0, w0, wd0, k0 = fn(b)
                p1, w1, wd1, k1 = nfn(na)
                s = 0.55
                ui = smoother(np.clip(u * (1 + s) - s * np.linspace(1, 0, NL), 0, 1))[:, None]
                # arc the travelling line slightly toward the lens so it reads as a gesture
                lift = np.sin(math.pi * ui) * np.array([0.0, -60.0, -120.0])
                p = p0 + (p1 - p0) * ui + lift
                w = w0 + (w1 - w0) * ui[:, 0]
                um = smoother(u)
                return p, w, wd0 + (wd1 - wd0) * um, k0 + (k1 - k0) * um, self.gain(t) * (1 + 0.25 * math.sin(math.pi * u))
        p, w, wd, kn = st[0][0](0.0)
        return p, w, wd, kn, 0.0


# ----------------------------------------------------------------------------
# The universe: thousands of floating videos
# ----------------------------------------------------------------------------
N_CARDS = 2500
N_TEX = 48


class Universe:
    def __init__(self, rig, line, heroes, seed=7):
        rng = np.random.default_rng(seed)
        # camera path xy as a function of z (for clearing a flight tube)
        ts = np.linspace(0, DURATION, 800)
        cp = np.array([rig.at(t).pos for t in ts])
        order = np.argsort(cp[:, 2])
        zs, xs, ys = cp[order, 2], cp[order, 0], cp[order, 1]
        boxes = []
        for h in heroes:
            c = h.corners(20.0) if h.name in ("logo", "wordmark", "last") else h.corners(5.0)
            x0, y0 = c[:, 0].min(), c[:, 1].min()
            x1, y1 = c[:, 0].max(), c[:, 1].max()
            z = c[:, 2].mean()
            boxes.append((x0 - 260, y0 - 220, x1 + 260, y1 + 220, z - 2900, z + 1500))
        boxes.append((-1700, -900, 1700, 900, 19000, 30000))  # keep the end card clean
        boxes.append((-900, -650, 900, 650, -800, 1900))       # icon + search stay clean
        P, n = [], 0
        while len(P) < N_CARDS and n < N_CARDS * 30:
            n += 1
            z = rng.uniform(2300, 30000)
            x = np.sign(rng.uniform(-1, 1)) * rng.uniform(0, 1) ** 1.15 * 8000
            y = np.sign(rng.uniform(-1, 1)) * rng.uniform(0, 1) ** 1.15 * 4800
            cx, cy = np.interp(z, zs, xs), np.interp(z, zs, ys)
            if math.hypot(x - cx, (y - cy) * 1.3) < 470:
                continue
            if any(b[0] < x < b[2] and b[1] < y < b[3] and b[4] < z < b[5] for b in boxes):
                continue
            P.append((x, y, z))
        self.pos = np.array(P)
        m = len(self.pos)
        self.scale = rng.uniform(0.8, 1.2, m)
        self.rot = np.stack([rng.normal(0, 14, m), rng.normal(0, 6, m), rng.normal(0, 3, m)], 1)
        self.tex = rng.integers(0, N_TEX, m)
        self.watch = np.where(rng.uniform(0, 1, m) < 0.78, rng.uniform(0.12, 1.0, m), 0.0)
        self.phase = rng.uniform(0, 2 * math.pi, (m, 3))
        self.freq = rng.uniform(0.12, 0.3, m)
        # ignition: the comet head lights the watched-bars as it passes
        hz = np.maximum.accumulate(line.head_p[:, 2])
        ht = line.head_t
        hxy = line.head_p[:, :2]
        z = self.pos[:, 2]
        t_ign = np.interp(z, hz, ht)
        hx = np.interp(z, hz, hxy[:, 0])
        hy = np.interp(z, hz, hxy[:, 1])
        lateral = np.hypot(self.pos[:, 0] - hx, self.pos[:, 1] - hy)
        t_ign = t_ign + lateral / 4200.0
        t_ign = np.where(z < hz[0], ht[0] + (hz[0] - z) / 3000 + lateral / 4200, t_ign)
        t_ign = np.where(z > hz[-1], ht[-1] + (z - hz[-1]) / 9000 + lateral / 6000, t_ign)
        self.t_ign = t_ign
        # local card geometry
        w = A.CARD_W * self.scale
        h = A.CARD_H * self.scale
        self.w, self.h = w, h

    def frame_geometry(self, t):
        """Card frames at time t: origin TL (m,3), x axis (m,3) per unit u, y axis per unit v."""
        drift = np.stack([np.sin(self.freq * t + self.phase[:, 0]) * 26,
                          np.sin(self.freq * 0.8 * t + self.phase[:, 1]) * 20,
                          np.sin(self.freq * 0.6 * t + self.phase[:, 2]) * 30], 1)
        c = self.pos + drift
        rot = self.rot + np.stack([np.sin(self.freq * t + self.phase[:, 1]) * 4,
                                   np.sin(self.freq * t + self.phase[:, 2]) * 2, np.zeros(len(c))], 1)
        yaw, pitch, roll = np.radians(rot).T
        cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
        # M = Ry @ Rx @ Rz ; columns are local axes
        ex = np.stack([cy * cr + sy * sp * sr, cp * sr, -sy * cr + cy * sp * sr], 1)
        ey = np.stack([-cy * sr + sy * sp * cr, cp * cr, sy * sr + cy * sp * cr], 1)
        s = self.scale[:, None]
        tl = c - ex * (A.CARD_W * s / 2) - ey * (A.CARD_H * s / 2)
        return tl, ex * s, ey * s

    def watched(self, t):
        k = ease_out_cubic(np.clip((t - self.t_ign) / 0.5, 0, 1))
        return self.watch * k, k


class World:
    def __init__(self):
        self.rig = CameraRig(CAM_KEYS, FOCAL, WIDTH, HEIGHT)
        self.heroes = build_heroes()
        self.line = Line(self.rig)
        self.universe = Universe(self.rig, self.line, self.heroes)

    def universe_gain(self, t):
        return 0.62 * (0.55 + 0.45 * smooth(lin(t, 2.5, 4.8))) * (1 - 0.55 * smooth(lin(t, 15.0, 17.6)))
