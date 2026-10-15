"""Fonts and texture painters (Pillow). Every texture is a premultiplied float32 RGBA array."""
import glob
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ----------------------------------------------------------------------------
# Palette (YouTube dark UI, rebuilt by eye)
# ----------------------------------------------------------------------------
RED = (255, 0, 0)
STAGE = (8, 8, 8)
BG = (15, 15, 15)
SURF1 = (33, 33, 33)
SURF2 = (39, 39, 39)
BORDER = (48, 48, 48)
TEXT1 = (241, 241, 241)
TEXT2 = (170, 170, 170)
WHITE = (255, 255, 255)

SS = 2  # supersampling factor for anti-aliased shapes

# ----------------------------------------------------------------------------
# Fonts: SF Pro if present, Inter as the stand-in.
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SF_DIRS = [os.path.join(_HERE, "fonts"), os.path.expanduser("~/Library/Fonts"), "/Library/Fonts",
            "/System/Library/Fonts", "/usr/share/fonts", os.path.expanduser("~/.fonts")]
_WEIGHTS = {"regular": ("Regular", "Regular"), "medium": ("Medium", "Medium"),
            "semibold": ("Semibold", "SemiBold"), "bold": ("Bold", "Bold"), "heavy": ("Heavy", "ExtraBold")}
_font_cache = {}


def _find(patterns):
    for d in _SF_DIRS:
        for p in patterns:
            hits = glob.glob(os.path.join(d, "**", p), recursive=True)
            if hits:
                return hits[0]
    return None


def font_path(weight="regular", display=False):
    sf_w, inter_w = _WEIGHTS[weight]
    fam = "Display" if display else "Text"
    p = _find([f"SF-Pro-{fam}-{sf_w}.otf", f"SFPro{fam}-{sf_w}.otf"])
    if p:
        return p
    p = _find([f"InterDisplay-{inter_w}.otf" if display else f"Inter-{inter_w}.otf"])
    if p:
        return p
    return _find(["DejaVuSans-Bold.ttf" if weight in ("bold", "heavy", "semibold") else "DejaVuSans.ttf"])


def font(size, weight="regular", display=False):
    key = (int(round(size)), weight, display)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(font_path(weight, display), max(1, key[0]))
    return _font_cache[key]


def font_name():
    return os.path.basename(font_path("semibold", True))


# ----------------------------------------------------------------------------
# Canvas in world units
# ----------------------------------------------------------------------------

class Canvas:
    """RGBA drawing surface addressed in world units (u), rasterised at ppu * SS."""

    def __init__(self, w, h, ppu=1.0, bg=(0, 0, 0, 0), ss=SS):
        self.w, self.h, self.ppu, self.ss = w, h, ppu, ss
        self.k = ppu * ss
        self.img = Image.new("RGBA", (max(1, round(w * self.k)), max(1, round(h * self.k))), bg)
        self.d = ImageDraw.Draw(self.img)

    def _p(self, *xy):
        return [v * self.k for v in xy]

    def rrect(self, x0, y0, x1, y1, r, fill=None, outline=None, width=0):
        self.d.rounded_rectangle(self._p(x0, y0, x1, y1), radius=r * self.k, fill=fill,
                                 outline=outline, width=int(round(width * self.k)))

    def rect(self, x0, y0, x1, y1, fill):
        self.d.rectangle(self._p(x0, y0, x1, y1), fill=fill)

    def circle(self, cx, cy, r, fill=None, outline=None, width=0):
        self.d.ellipse(self._p(cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline,
                       width=int(round(width * self.k)))

    def poly(self, pts, fill):
        self.d.polygon([(x * self.k, y * self.k) for x, y in pts], fill=fill)

    def line(self, pts, fill, width):
        self.d.line([(x * self.k, y * self.k) for x, y in pts], fill=fill, width=max(1, int(round(width * self.k))))

    def text(self, x, y, s, size, fill, weight="regular", display=False, anchor="la"):
        self.d.text((x * self.k, y * self.k), s, font=font(size * self.k, weight, display), fill=fill, anchor=anchor)

    def text_w(self, s, size, weight="regular", display=False):
        return font(size * self.k, weight, display).getlength(s) / self.k

    def paste(self, other, x, y):
        self.img.alpha_composite(other.img if isinstance(other, Canvas) else other,
                                 (int(round(x * self.k)), int(round(y * self.k))))

    def magnifier(self, cx, cy, s, color, width):
        r = s * 0.36
        self.circle(cx - s * 0.1, cy - s * 0.1, r, outline=color, width=width)
        a = (cx - s * 0.1 + r * 0.72, cy - s * 0.1 + r * 0.72)
        self.line([a, (cx + s * 0.42, cy + s * 0.42)], color, width)

    def finish(self):
        im = self.img
        if self.ss != 1:
            im = im.resize((max(1, round(self.w * self.ppu)), max(1, round(self.h * self.ppu))), Image.LANCZOS)
        return premult(im)


def premult(im):
    a = np.asarray(im, np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return np.ascontiguousarray(a)


def rrect_mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return m


def gradient(w, h, c0, c1, angle=90.0, curve=1.0):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle)
    proj = (xs / max(w - 1, 1)) * math.cos(a) + (ys / max(h - 1, 1)) * math.sin(a)
    proj = (proj - proj.min()) / max(proj.max() - proj.min(), 1e-6)
    proj = proj ** curve
    c0, c1 = np.array(c0, np.float32), np.array(c1, np.float32)
    rgb = c0 + (c1 - c0) * proj[..., None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def hexc(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


# ----------------------------------------------------------------------------
# Content for the universe of thumbnails (fictional)
# ----------------------------------------------------------------------------
TITLES = [
    "How to Bake Sourdough at Home", "How to Tie a Bowline Knot", "How to Fold a Fitted Sheet",
    "How to Fix a Flat Bike Tire", "How to Play Your First Chord", "How to Draw Hands (Finally)",
    "How to Sharpen a Kitchen Knife", "How to Brew Pour-Over Coffee", "How to Change a Car Battery",
    "How to Start Running Again", "How to Grow Basil Indoors", "How to Solve a Cube in 2 Minutes",
    "How to Paint a Sunset Sky", "How to Edit Your First Video", "How to Patch Drywall",
    "How to Make Fresh Pasta", "How to Juggle Three Balls", "How to Code a Game in a Weekend",
    "How to Repot a Monstera", "How to Do a Kickflip", "How to Fold a Paper Crane",
    "How to Make Cold Brew", "How to Build a Bookshelf", "How to Shoot in Low Light",
    "How to Learn Any Language", "How to Knit Your First Scarf", "How to Clean Cast Iron",
    "How to Read Sheet Music", "How to Moonwalk", "How to Mix Your First Track",
    "How to Whistle Really Loud", "How to Make Ramen From Scratch", "How to Train Your Puppy",
    "How to Unclog Any Drain", "How to Write a Song Tonight", "How to Do a Handstand",
    "How to Cut Your Own Hair", "How to Plant a Tree", "How to Make a Simple Budget",
    "How to Animate a Bouncing Ball", "How to Fillet a Fish", "How to Pitch a Tent in the Rain",
    "How to Make Soy Candles", "How to Restore Old Photos", "How to Sew On a Button",
    "How to Make Sushi Rice", "How to Ollie", "How to Change a Tire",
]
CHANNELS = ["Hands On", "Kitchen Lab", "Slow Workshop", "Fix It Friday", "Mia Learns", "Studio Twelve",
            "The Patient Cook", "Wrench & Co", "Small Space Garden", "Neon Lessons", "Half Hour Hobby",
            "Rough Draft", "Practice Room", "Field Notes", "Every Day Maker"]
WORDS = ["EASY", "5 MIN", "STEP 1", "DIY", "PRO TIP", "BEGINNER", "IN 1 DAY", "NO TOOLS", "100%",
         "FIXED", "SIMPLE", "FIRST TRY", "AT HOME", "3 STEPS"]
# Moody, cool / warm-neutral palettes. Deliberately no reds: red belongs to the line.
PALETTES = [("#14283a", "#6fa3c7"), ("#22242f", "#8d99ae"), ("#1e3024", "#9fbf7a"), ("#2c2418", "#e0a458"),
            ("#1a2238", "#f2c14e"), ("#251a33", "#b48ad6"), ("#0f2c31", "#5fd3c9"), ("#2f281f", "#d9c5a0"),
            ("#1a1a1a", "#d8d8d8"), ("#101a30", "#fca311"), ("#0b2d2b", "#e9d8a6"), ("#24273a", "#93b7be"),
            ("#1b2a1f", "#cfe3a5"), ("#2a2030", "#f0b6a0")]

CARD_W, CARD_H = 512, 404
THUMB_H = 288


def thumbnail_art(w, h, seed, word=True):
    rnd = random.Random(seed)
    c0, c1 = PALETTES[seed % len(PALETTES)]
    c0, c1 = hexc(c0), hexc(c1)
    im = gradient(w, h, c0, tuple(int(a * 0.55 + b * 0.45) for a, b in zip(c0, c1)), rnd.uniform(20, 160))
    d = ImageDraw.Draw(im, "RGBA")
    motif = rnd.randrange(6)
    acc = c1 + (255,)
    soft = c1 + (90,)
    if motif == 0:  # sun over horizon
        r = h * rnd.uniform(0.25, 0.4)
        cx, cy = w * rnd.uniform(0.55, 0.8), h * 0.55
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=acc)
        d.rectangle([0, h * 0.62, w, h], fill=c0 + (235,))
    elif motif == 1:  # stacked blocks
        x = w * 0.5
        for i in range(rnd.randint(3, 5)):
            bw, bh = w * rnd.uniform(0.18, 0.3), h * rnd.uniform(0.12, 0.2)
            y = h * 0.9 - (i + 1) * bh
            d.rounded_rectangle([x - bw / 2 + rnd.uniform(-20, 20), y, x + bw / 2, y + bh * 0.92], 8,
                                fill=acc if i % 2 == 0 else soft)
    elif motif == 2:  # rings
        cx, cy = w * rnd.uniform(0.55, 0.75), h * 0.5
        for i in range(5, 0, -1):
            r = h * 0.09 * i
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=c1 + (60 + 35 * (5 - i),), width=max(3, w // 90))
    elif motif == 3:  # mountains
        pts = [(0, h)]
        for i in range(6):
            pts.append((w * i / 5, h * rnd.uniform(0.35, 0.75)))
        pts.append((w, h))
        d.polygon(pts, fill=c1 + (170,))
        d.polygon([(0, h), (w * 0.3, h * 0.6), (w * 0.7, h * 0.8), (w, h * 0.65), (w, h)], fill=c0 + (230,))
    elif motif == 4:  # tool / object silhouette
        cx, cy = w * 0.66, h * 0.52
        d.rounded_rectangle([cx - w * 0.2, cy - h * 0.08, cx + w * 0.2, cy + h * 0.08], 12, fill=acc)
        d.rectangle([cx - w * 0.03, cy + h * 0.08, cx + w * 0.03, cy + h * 0.4], fill=soft)
    else:  # diagonal stripes
        for i in range(-4, 12):
            x = i * w / 8
            d.polygon([(x, h), (x + w * 0.06, h), (x + w * 0.36, 0), (x + w * 0.3, 0)], fill=c1 + (40,))
        r = h * 0.22
        d.ellipse([w * 0.62 - r, h * 0.5 - r, w * 0.62 + r, h * 0.5 + r], fill=acc)
    if word:
        s = WORDS[seed % len(WORDS)]
        f = font(h * 0.2, "heavy", True)
        x, y = w * 0.06, h * 0.08
        d.text((x + 3, y + 4), s, font=f, fill=(0, 0, 0, 110))
        d.text((x, y), s, font=f, fill=(255, 255, 255, 245))
    # gentle vignette
    v = Image.new("L", (w, h), 0)
    ImageDraw.Draw(v).ellipse([-w * 0.25, -h * 0.35, w * 1.25, h * 1.35], fill=255)
    v = v.filter(ImageFilter.GaussianBlur(h * 0.12))
    dark = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    im = Image.composite(im, Image.blend(im, dark, 0.45), v)
    return im


def card_texture(seed):
    """A YouTube grid card: 16:9 thumbnail, duration badge, avatar, two-line title, meta."""
    rnd = random.Random(seed * 7 + 3)
    c = Canvas(CARD_W, CARD_H, 1.0)
    k = c.k
    art = thumbnail_art(int(CARD_W * k), int(THUMB_H * k), seed)
    mask = rrect_mask(art.width, art.height, int(16 * k))
    c.img.paste(art, (0, 0), mask)
    dur = f"{rnd.randint(2, 48)}:{rnd.randint(0, 59):02d}"
    tw = c.text_w(dur, 20, "medium") + 14
    c.rrect(CARD_W - 10 - tw, THUMB_H - 38, CARD_W - 10, THUMB_H - 12, 5, fill=(0, 0, 0, 205))
    c.text(CARD_W - 10 - tw / 2, THUMB_H - 25, dur, 20, WHITE, "medium", anchor="mm")
    pal = hexc(PALETTES[(seed * 3 + 1) % len(PALETTES)][1])
    c.circle(22, THUMB_H + 36, 20, fill=pal + (255,))
    title = TITLES[seed % len(TITLES)]
    words, lines, cur = title.split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if c.text_w(trial, 26, "medium") > CARD_W - 60 and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    lines.append(cur)
    for i, ln in enumerate(lines[:2]):
        c.text(56, THUMB_H + 16 + i * 33, ln, 26, TEXT1, "medium")
    ch = CHANNELS[seed % len(CHANNELS)]
    meta = f"{ch}  ·  {rnd.choice(['12K', '86K', '340K', '1.2M', '2.4M', '7.9M'])} views"
    c.text(56, THUMB_H + 16 + min(len(lines), 2) * 33 + 6, meta, 21, TEXT2)
    return c.finish()


# ----------------------------------------------------------------------------
# Hero textures
# ----------------------------------------------------------------------------

def icon_texture(w, h, ppu, fill=1.0, buffered=1.0, tri=1.0, tri_scale=1.0, base_alpha=1.0):
    """The play icon. fill / buffered are left-to-right loading fractions (logo loading like a video)."""
    c = Canvas(w, h, ppu)
    r = h * 0.235
    if base_alpha > 0:
        c.rrect(0, 0, w, h, r, fill=(255, 255, 255, int(16 * base_alpha)))
    full = Canvas(w, h, ppu)
    full.rrect(0, 0, w, h, r, fill=RED + (255,))
    buf = Canvas(w, h, ppu)
    buf.rrect(0, 0, w, h, r, fill=(255, 255, 255, 52))
    W = c.img.width
    if buffered > 0:
        bx = int(W * min(1, buffered))
        if bx > 0:
            c.img.alpha_composite(buf.img.crop((0, 0, bx, c.img.height)), (0, 0))
    if fill > 0:
        fx = int(W * min(1, fill))
        if fx > 0:
            c.img.alpha_composite(full.img.crop((0, 0, fx, c.img.height)), (0, 0))
    if tri > 0:
        tw = w * 0.30 * tri_scale
        th = tw * 1.15
        cx, cy = w * 0.515, h * 0.5
        c.poly([(cx - tw * 0.42, cy - th / 2), (cx - tw * 0.42, cy + th / 2), (cx + tw * 0.58, cy)],
               (255, 255, 255, int(255 * tri)))
    return c.finish()


def search_texture(w, h, ppu, typed, caret):
    c = Canvas(w, h, ppu)
    bw = 150
    r = h / 2
    c.rrect(0, 0, w, h, r, fill=SURF2 + (255,), outline=BORDER + (255,), width=2)
    c.rrect(1, 1, w - bw, h - 1, r, fill=(18, 18, 18, 255), outline=(28, 28, 28, 255), width=2)
    c.rect(w - bw - r, 1, w - bw, h - 1, fill=(18, 18, 18, 255))
    c.rect(w - bw, 2, w - bw + 2, h - 2, fill=BORDER + (255,))
    c.magnifier(w - bw / 2, h / 2, 40, TEXT1 + (255,), 4)
    c.magnifier(52, h / 2, 34, TEXT2 + (255,), 3.5)
    size = 50
    x0 = 96
    c.text(x0, h / 2, typed, size, TEXT1, "regular", anchor="lm")
    if caret:
        x = x0 + c.text_w(typed, size) + 4
        c.rect(x, h / 2 - 28, x + 3, h / 2 + 28, fill=TEXT1 + (255,))
    return c.finish()


def panel_texture(w, h, ppu):
    c = Canvas(w, h, ppu)
    c.rrect(0, 0, w, h, 22, fill=SURF1 + (255,), outline=(44, 44, 44, 255), width=1.5)
    return c.finish()


def suggestion_texture(w, h, ppu, prefix, completion, hilite=False):
    c = Canvas(w, h, ppu)
    if hilite:
        c.rrect(8, 0, w - 8, h, 10, fill=(255, 255, 255, 22))
    c.magnifier(44, h / 2, 28, TEXT2 + (255,), 3)
    size = 36
    c.text(88, h / 2, prefix, size, TEXT1, "regular", anchor="lm")
    x = 88 + c.text_w(prefix, size)
    c.text(x, h / 2, completion, size, TEXT1, "bold", anchor="lm")
    return c.finish()


def player_texture(w, h, ppu, art, time_text, chapters=None, chapter_name=None, paused=False):
    """Video player: frame, bottom scrim, progress track (chapters as gaps) and controls."""
    c = Canvas(w, h, ppu)
    k = c.k
    frame = art.resize((int(w * k), int(h * k)), Image.LANCZOS)
    c.img.paste(frame, (0, 0), rrect_mask(frame.width, frame.height, int(18 * k)))
    sh = int(h * 0.32 * k)
    scrim = np.zeros((sh, frame.width, 4), np.uint8)
    scrim[..., 3] = (np.linspace(0, 1, sh) ** 1.6 * 220).astype(np.uint8)[:, None]
    scrim_im = Image.fromarray(scrim, "RGBA")
    bottom = Image.new("RGBA", c.img.size, (0, 0, 0, 0))
    bottom.paste(scrim_im, (0, c.img.height - sh))
    bottom.putalpha(Image.fromarray(np.minimum(np.asarray(bottom)[..., 3],
                                               np.asarray(rrect_mask(*c.img.size, int(18 * k))))))
    c.img.alpha_composite(bottom)
    pad = 20
    ty = h - 54
    chapters = chapters or [0.0, 1.0]
    for a, b in zip(chapters[:-1], chapters[1:]):
        x0 = pad + (w - 2 * pad) * a + (2.5 if a > 0 else 0)
        x1 = pad + (w - 2 * pad) * b - (2.5 if b < 1 else 0)
        c.rect(x0, ty - 2.5, x1, ty + 2.5, fill=(255, 255, 255, 70))
    cy = h - 26
    if paused:
        c.poly([(pad + 6, cy - 11), (pad + 6, cy + 11), (pad + 25, cy)], TEXT1 + (255,))
    else:
        c.rect(pad + 6, cy - 11, pad + 12, cy + 11, fill=TEXT1 + (255,))
        c.rect(pad + 18, cy - 11, pad + 24, cy + 11, fill=TEXT1 + (255,))
    c.poly([(pad + 48, cy - 5), (pad + 54, cy - 5), (pad + 62, cy - 12), (pad + 62, cy + 12),
            (pad + 54, cy + 5), (pad + 48, cy + 5)], TEXT1 + (255,))
    label = time_text + (f"   ·   {chapter_name}  ›" if chapter_name else "")
    c.text(pad + 84, cy, label, 20, TEXT1, "medium", anchor="lm")
    c.circle(w - pad - 70, cy, 10, outline=TEXT1 + (255,), width=3)
    for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        x, y = w - pad - 22 + dx * 9, cy + dy * 9
        c.line([(x, y), (x - dx * 6, y)], TEXT1 + (255,), 3)
        c.line([(x, y), (x, y - dy * 6)], TEXT1 + (255,), 3)
    return c.finish()


def frame_art(kind, w, h):
    """Procedural 'video' frames for the hero players."""
    if kind == "pasta":
        im = gradient(w, h, (62, 44, 28), (196, 160, 112), 115, 0.9)
        d = ImageDraw.Draw(im, "RGBA")
        d.ellipse([w * 0.18, h * 0.18, w * 0.78, h * 1.05], fill=(236, 226, 205, 255))
        d.ellipse([w * 0.25, h * 0.28, w * 0.71, h * 0.95], fill=(222, 208, 182, 255))
        rnd = random.Random(4)
        for _ in range(26):
            x, y = rnd.uniform(w * 0.3, w * 0.66), rnd.uniform(h * 0.38, h * 0.85)
            d.arc([x - 40, y - 14, x + 40, y + 14], rnd.uniform(0, 180), rnd.uniform(200, 360),
                  fill=(232, 196, 120, 255), width=7)
        for _ in range(240):
            x, y = rnd.uniform(0, w), rnd.uniform(0, h)
            d.ellipse([x, y, x + 3, y + 3], fill=(255, 250, 240, 90))
    elif kind == "stool":
        im = gradient(w, h, (34, 30, 26), (120, 96, 70), 75, 1.1)
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([0, h * 0.78, w, h], fill=(70, 52, 36, 255))
        for i in range(9):
            d.line([(0, h * 0.8 + i * 14), (w, h * 0.79 + i * 15)], fill=(90, 68, 46, 120), width=3)
        sx, sy = w * 0.5, h * 0.36
        d.rounded_rectangle([sx - w * 0.17, sy - 20, sx + w * 0.17, sy + 22], 14, fill=(214, 168, 112, 255))
        for dx, ang in ((-0.12, -0.05), (0.12, 0.05), (-0.05, -0.02), (0.05, 0.02)):
            x0 = sx + w * dx
            d.polygon([(x0 - 9, sy + 20), (x0 + 9, sy + 20), (x0 + 9 + w * ang * 2, h * 0.8),
                       (x0 - 9 + w * ang * 2, h * 0.8)], fill=(186, 140, 90, 255))
        d.rectangle([sx - w * 0.13, sy + h * 0.25, sx + w * 0.13, sy + h * 0.25 + 10], fill=(170, 126, 80, 255))
        d.ellipse([w * 0.72, h * 0.08, w * 0.98, h * 0.5], fill=(255, 225, 170, 26))
    else:  # creator selfie-cam
        im = gradient(w, h, (20, 32, 40), (64, 92, 104), 60, 1.0)
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([w * 0.06, h * 0.12, w * 0.3, h * 0.56], fill=(40, 58, 66, 255))
        for i in range(4):
            d.rectangle([w * 0.08, h * (0.16 + i * 0.1), w * 0.28, h * (0.16 + i * 0.1) + 8],
                        fill=(110, 140, 150, 120))
        d.ellipse([w * 0.74, h * 0.1, w * 0.96, h * 0.5], fill=(240, 220, 170, 40))
        d.ellipse([w * 0.42, h * 0.2, w * 0.6, h * 0.58], fill=(26, 34, 40, 255))
        d.rounded_rectangle([w * 0.3, h * 0.55, w * 0.72, h * 1.3], int(h * 0.2), fill=(26, 34, 40, 255))
    v = Image.new("L", (w, h), 0)
    ImageDraw.Draw(v).ellipse([-w * 0.2, -h * 0.3, w * 1.2, h * 1.3], fill=255)
    v = v.filter(ImageFilter.GaussianBlur(h * 0.1))
    dark = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    return Image.composite(im, Image.blend(im, dark, 0.5), v)


def watch_info_texture(w, h, ppu, title, channel, subs):
    c = Canvas(w, h, ppu)
    c.text(4, 8, title, 40, TEXT1, "semibold", display=True)
    y = 96
    c.circle(30, y, 26, fill=(224, 164, 88, 255))
    c.text(72, y - 14, channel, 26, TEXT1, "semibold", anchor="lm")
    c.text(72, y + 16, subs, 21, TEXT2, anchor="lm")
    x = 72 + max(c.text_w(channel, 26, "semibold"), c.text_w(subs, 21)) + 28
    pw = c.text_w("Subscribe", 24, "semibold") + 44
    c.rrect(x, y - 24, x + pw, y + 24, 24, fill=(241, 241, 241, 255))
    c.text(x + pw / 2, y, "Subscribe", 24, (15, 15, 15), "semibold", anchor="mm")
    lx = w - 240
    c.rrect(lx, y - 24, lx + 236, y + 24, 24, fill=SURF2 + (255,))
    c.text(lx + 118, y, "12K   |   Share", 22, TEXT1, "medium", anchor="mm")
    return c.finish()


def chapter_info_texture(w, h, ppu, title, idx, total, name):
    c = Canvas(w, h, ppu)
    c.text(4, 8, title, 40, TEXT1, "semibold", display=True)
    c.rrect(4, 74, w - 4, 136, 14, fill=SURF2 + (255,))
    c.text(28, 105, f"Chapter {idx} of {total}", 23, TEXT2, "medium", anchor="lm")
    x = 28 + c.text_w(f"Chapter {idx} of {total}", 23, "medium") + 18
    c.text(x, 105, name, 25, TEXT1, "semibold", anchor="lm")
    return c.finish()


def super_texture(lines, size, ppu, align="left", color=TEXT1, weight="semibold"):
    lh = size * 1.08
    tmp = Canvas(10, 10, ppu)
    w = max(tmp.text_w(s, size, weight, True) for s in lines) + size * 0.2
    h = lh * len(lines) + size * 0.35
    c = Canvas(w, h, ppu)
    for i, s in enumerate(lines):
        x = {"left": size * 0.05, "right": w - size * 0.05, "center": w / 2}[align]
        anchor = {"left": "ls", "right": "rs", "center": "ms"}[align]
        c.text(x, size * 0.95 + i * lh, s, size, color, weight, True, anchor)
    return c.finish(), (w, h)


def text_size(s, size, weight="semibold", display=True):
    f = font(size * 8, weight, display)
    x0, y0, x1, y1 = f.getbbox(s, anchor="ls")
    return f.getlength(s) / 8, -y0 / 8, y1 / 8


def viewfinder_texture(w, h, ppu, art, rec_on, rec_time, dot_on, disk, square, status):
    c = Canvas(w, h, ppu)
    k = c.k
    frame = art.resize((int(w * k), int(h * k)), Image.LANCZOS)
    c.img.paste(frame, (0, 0), rrect_mask(frame.width, frame.height, int(18 * k)))
    L, m = 46, 28
    for (x, y, dx, dy) in ((m, m, 1, 1), (w - m, m, -1, 1), (w - m, h - m, -1, -1), (m, h - m, 1, -1)):
        c.line([(x, y + dy * L), (x, y), (x + dx * L, y)], (255, 255, 255, 200), 3)
    if rec_on:
        c.rrect(m + 16, m + 14, m + 210, m + 58, 10, fill=(0, 0, 0, 140))
        if dot_on:
            c.circle(m + 40, m + 36, 9, fill=RED + (255,))
        c.text(m + 60, m + 36, f"REC  {rec_time}", 24, TEXT1, "semibold", anchor="lm")
    if status:
        tw = c.text_w(status, 22, "medium") + 30
        c.rrect(w - m - 16 - tw, m + 14, w - m - 16, m + 58, 10, fill=(0, 0, 0, 140))
        c.text(w - m - 16 - tw / 2, m + 36, status, 22, TEXT1, "medium", anchor="mm")
    # shutter button interior (the ring itself is the line)
    bx, by = w / 2, h - 70
    c.circle(bx, by, 46, fill=(0, 0, 0, 90))
    if disk > 0:
        if square > 0:
            s = 34 - 16 * square
            c.rrect(bx - s, by - s, bx + s, by + s, s * (1 - 0.7 * square), fill=RED + (int(255 * disk),))
        else:
            r = 35 * disk
            c.circle(bx, by, r, fill=RED + (255,))
    return c.finish()


def upload_info_texture(w, h, ppu, title, label, pct_text):
    c = Canvas(w, h, ppu)
    c.text(4, 44, title, 30, TEXT1, "semibold", display=True, anchor="lm")
    c.text(w - 4, 44, pct_text, 26, TEXT2, "medium", anchor="rm")
    if label:
        c.text(4, 86, label, 22, TEXT2, anchor="lm")
    return c.finish()


def track_texture(w, h, ppu):
    c = Canvas(w, h, ppu)
    c.rrect(0, 0, w, h, h / 2, fill=(255, 255, 255, 46))
    return c.finish()
