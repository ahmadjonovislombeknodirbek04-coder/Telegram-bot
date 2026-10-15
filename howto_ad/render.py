"""Frame renderer + encoder for "How to Anything".

    python -m howto_ad.render                     # full film -> out/how_to_anything.mp4
    python -m howto_ad.render --stills 0.5,3,9    # PNG stills at given seconds
    python -m howto_ad.render --scale 0.5 --start 6 --end 8   # fast low-res preview
"""
import argparse
import math
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

from . import assets as A
from .core import lin, smooth
from .world import DURATION, FPS, HEIGHT, MAIN_DUR, WIDTH, World

cv2.setNumThreads(1)

SHUTTER = 0.9 / FPS        # heavy motion blur: 324 degree shutter
APERTURE = 7.0             # world units; small = deep depth of field
U_SCALE = 0.5              # universe layer is rendered at half resolution
BLUR_LEVELS = [0.0, 1.2, 2.4, 4.2, 7.0]
STAGE = np.array(A.STAGE, np.float32) / 255
RED = np.array([1.0, 0.0, 0.0], np.float32)

_W = None  # per-process world


class Renderer:
    def __init__(self, scale=1.0):
        self.world = World()
        self.scale = scale
        self.W, self.H = int(WIDTH * scale), int(HEIGHT * scale)
        self.UW, self.UH = int(self.W * U_SCALE), int(self.H * U_SCALE)
        self.tex_cache = {}
        self.card_mips = []
        self.card_avg = []
        for i in range(48):
            tex = A.card_texture(i)
            mips = [tex]
            while mips[-1].shape[1] > 40:
                mips.append(cv2.pyrDown(mips[-1]))
            self.card_mips.append(mips)
            thumb = tex[: A.THUMB_H]
            self.card_avg.append(thumb.reshape(-1, 4).mean(0))
        self.card_avg = np.array(self.card_avg, np.float32)
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        r = np.hypot((xx - self.W / 2) / (self.W / 2), (yy - self.H / 2) / (self.H / 2))
        self.vignette = (1 - 0.2 * np.clip(r - 0.35, 0, None) ** 1.6)[..., None].astype(np.float32)
        self.yy, self.xx = yy, xx

    # ------------------------------------------------------------------
    def cam(self, t, k=1.0):
        return self.world.rig.at(t).scaled(self.scale * k)

    def sub_times(self, t, n):
        return t + ((np.arange(n) + 0.5) / n - 0.5) * SHUTTER

    def motion_px(self, t, depth, k=1.0):
        c0, c1 = self.cam(t - SHUTTER / 2, k), self.cam(t + SHUTTER / 2, k)
        pts = []
        for sx in (-0.9, 0, 0.9):
            for sy in (-0.9, 0, 0.9):
                dirc = np.array([sx * c0.cx / c0.f, sy * c0.cy / c0.f, 1.0]) * depth
                pts.append(c0.pos + c0.R.T @ dirc)
        pts = np.array(pts)
        a, _ = c0.project(pts)
        b, _ = c1.project(pts)
        d = np.linalg.norm(a - b, axis=1)
        return float(np.nanmax(d)) if np.isfinite(d).any() else 50.0

    # ------------------------------------------------------------------
    # Universe of cards (half-res, depth-of-field layers, temporal samples)
    # ------------------------------------------------------------------
    def render_universe(self, t, gain):
        U = self.world.universe
        cam = self.cam(t, U_SCALE)
        tl, ex, ey = U.frame_geometry(t)
        center = tl + ex * (A.CARD_W / 2) + ey * (A.CARD_H / 2)
        pc = cam.to_cam(center)
        z = pc[:, 2]
        lim_x = (cam.cx / cam.f) * 1.25
        lim_y = (cam.cy / cam.f) * 1.25
        vis = (z > 60) & (z < 30000) & (np.abs(pc[:, 0]) < lim_x * z + 600) & (np.abs(pc[:, 1]) < lim_y * z + 600)
        idx = np.nonzero(vis)[0]
        out = np.zeros((self.UH, self.UW, 4), np.float32)
        if len(idx) == 0:
            return out
        coc = APERTURE * cam.f * np.abs(1.0 / z[idx] - 1.0 / max(cam.focus, 1.0))
        sig = coc * 0.5
        lvl = np.abs(np.array(BLUR_LEVELS)[None, :] - sig[:, None]).argmin(1)
        far = z[idx] > cam.focus
        groups = []
        for L in range(len(BLUR_LEVELS) - 1, 0, -1):
            groups.append(idx[far & (lvl == L)])
        groups.append(idx[lvl == 0])
        for L in range(1, len(BLUR_LEVELS)):
            groups.append(idx[(~far) & (lvl == L)])
        sig_of = [BLUR_LEVELS[L] for L in range(len(BLUR_LEVELS) - 1, 0, -1)] + [0.0] + BLUR_LEVELS[1:]
        n = int(np.clip(math.ceil(self.motion_px(t, 900, U_SCALE) / 3.0), 1, 14))
        cams = [self.cam(ts, U_SCALE) for ts in self.sub_times(t, n)]
        watched, ign = U.watched(t)
        # sort far -> near within groups
        for g, s in zip(groups, sig_of):
            if len(g) == 0:
                continue
            g = g[np.argsort(-z[g])]
            acc = np.zeros_like(out)
            for c in cams:
                sub = np.zeros_like(out)
                self._draw_cards(sub, c, g, tl, ex, ey, watched, ign, gain)
                acc += sub
            acc /= n
            if s > 0.3:
                acc = cv2.GaussianBlur(acc, (0, 0), s * self.scale)
            out = acc + out * (1 - acc[..., 3:4])
        return out

    def _draw_cards(self, buf, cam, g, tl, ex, ey, watched, ign, gain):
        H, W = buf.shape[:2]
        CW, CH, TH = A.CARD_W, A.CARD_H, A.THUMB_H
        loc = np.array([[0, 0], [CW, 0], [CW, CH], [0, CH]], np.float32)
        corners = tl[g, None, :] + ex[g, None, :] * loc[None, :, 0:1] + ey[g, None, :] * loc[None, :, 1:2]
        uv, zc = cam.project(corners.reshape(-1, 3), near=30)
        uv = uv.reshape(-1, 4, 2)
        zc = zc.reshape(-1, 4)
        for j, ci in enumerate(g):
            q = uv[j]
            if not np.isfinite(q).all():
                continue
            x0, y0 = q.min(0)
            x1, y1 = q.max(0)
            if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
                continue
            zm = zc[j].mean()
            fog = math.exp(-max(0.0, zm - 1400) / 4300) * gain
            prox = smooth(lin(zm, 140, 650))
            if prox <= 0.01:
                continue
            size = max(x1 - x0, y1 - y0)
            if size > W * 3:
                continue
            ti = self.world.universe.tex[ci]
            if size < 36:
                c = self.card_avg[ti]
                col = (float(c[0] * fog * prox), float(c[1] * fog * prox), float(c[2] * fog * prox), float(c[3] * prox))
                th = q[0] + (q[3] - q[0]) * (TH / CH), q[1] + (q[2] - q[1]) * (TH / CH)
                poly = np.array([q[0], q[1], th[1], th[0]])
                cv2.fillConvexPoly(buf, np.round(poly * 16).astype(np.int32), col, cv2.LINE_8, 4)
                if size > 7:
                    a0 = q[0] + (q[3] - q[0]) * ((TH + 20) / CH)
                    b0 = q[1] + (q[2] - q[1]) * ((TH + 20) / CH)
                    a1 = q[0] + (q[3] - q[0]) * ((TH + 44) / CH)
                    b1 = q[1] + (q[2] - q[1]) * ((TH + 44) / CH)
                    m = 0.11
                    mid = [a0 + (b0 - a0) * m, a0 + (b0 - a0) * 0.8, a1 + (b1 - a1) * 0.8, a1 + (b1 - a1) * m]
                    v = 0.32 * fog * prox
                    cv2.fillConvexPoly(buf, np.round(np.array(mid) * 16).astype(np.int32), (v, v, v, 0.5 * prox),
                                       cv2.LINE_8, 4)
            else:
                mips = self.card_mips[ti]
                lv = int(np.clip(math.floor(math.log2(max(CW / max(size, 1), 1.0))), 0, len(mips) - 1))
                tex = mips[lv]
                th_, tw_ = tex.shape[:2]
                ix0, iy0 = max(int(x0), 0), max(int(y0), 0)
                ix1, iy1 = min(int(math.ceil(x1)) + 1, W), min(int(math.ceil(y1)) + 1, H)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                src = np.array([[0, 0], [tw_, 0], [tw_, th_], [0, th_]], np.float32)
                M = cv2.getPerspectiveTransform(src, (q - [ix0, iy0]).astype(np.float32))
                warp = cv2.warpPerspective(tex, M, (ix1 - ix0, iy1 - iy0), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT)
                warp *= prox
                warp[..., :3] *= fog
                roi = buf[iy0:iy1, ix0:ix1]
                roi *= 1 - warp[..., 3:4]
                roi += warp
            # watched bar (the line, everywhere)
            if ign[ci] > 0.002 and size > 5:
                f = watched[ci]
                bh = 6.0
                def lp(u, v):
                    return q[0] + (q[1] - q[0]) * (u / CW) + (q[3] - q[0]) * (v / CH)
                def quad(u0, u1):
                    return np.array([lp(u0, TH - bh), lp(u1, TH - bh), lp(u1, TH), lp(u0, TH)])
                ga = 0.55 * ign[ci] * prox
                gv = 0.5 * fog * ga
                cv2.fillConvexPoly(buf, np.round(quad(0, CW) * 16).astype(np.int32), (gv, gv, gv, ga), cv2.LINE_8, 4)
                if f > 0.004:
                    ra = prox * min(1.0, ign[ci] * 2)
                    rv = min(1.0, fog * 1.25) * ra
                    cv2.fillConvexPoly(buf, np.round(quad(0, CW * f) * 16).astype(np.int32), (rv, 0.0, 0.0, ra),
                                       cv2.LINE_8, 4)

    # ------------------------------------------------------------------
    # Heroes (full res)
    # ------------------------------------------------------------------
    def texture(self, hero, t, level):
        key = (hero.name, hero.key(t))
        mips = self.tex_cache.get(key)
        if mips is None:
            if len(self.tex_cache) > 400:
                self.tex_cache.clear()
            mips = [hero.paint(key[1])]
            self.tex_cache[key] = mips
        while len(mips) <= level and mips[-1].shape[1] > 16:
            mips.append(cv2.pyrDown(mips[-1]))
        return mips[min(level, len(mips) - 1)]

    def render_heroes(self, img, t, cam):
        hs = []
        for h in self.world.heroes:
            a = h.alpha(t)
            if a <= 0.003:
                continue
            c = h.corners(t)
            zc = cam.to_cam(c.mean(0))[2]
            prox = smooth(lin(zc, 160, 560))
            if prox * a <= 0.003:
                continue
            hs.append((zc, h, a * prox, c))
        if not hs:
            return
        mp = self.motion_px(t, 1400)
        n = int(np.clip(math.ceil(mp / 2.0), 1, 18))
        cams = [self.cam(ts) for ts in self.sub_times(t, n)]
        for zc, h, a, _ in sorted(hs, key=lambda x: -x[3][:, 2].mean()):
            quads = []
            for k, ts in enumerate(self.sub_times(t, n)):
                uv, zz = cams[k].project(h.corners(ts), near=20)
                if not np.isfinite(uv).all():
                    continue
                quads.append(uv)
            if not quads:
                continue
            allq = np.concatenate(quads)
            x0, y0 = np.floor(allq.min(0)).astype(int) - 2
            x1, y1 = np.ceil(allq.max(0)).astype(int) + 2
            if x1 < 0 or y1 < 0 or x0 >= self.W or y0 >= self.H:
                continue
            pad = 24
            x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
            x1, y1 = min(x1 + pad, self.W), min(y1 + pad, self.H)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            scr_w = np.linalg.norm(quads[0][1] - quads[0][0])
            tex_w = h.size[0] * h.ppu
            level = int(np.clip(math.floor(math.log2(max(tex_w / max(scr_w, 1) / 1.15, 1.0))), 0, 6))
            tex = self.texture(h, t, level)
            th_, tw_ = tex.shape[:2]
            src = np.array([[0, 0], [tw_, 0], [tw_, th_], [0, th_]], np.float32)
            acc = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
            for q in quads:
                M = cv2.getPerspectiveTransform(src, (q - [x0, y0]).astype(np.float32))
                acc += cv2.warpPerspective(tex, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT)
            acc *= a / len(quads)
            if not h.sharp:
                coc = APERTURE * cam.f * abs(1.0 / max(zc, 1) - 1.0 / max(cam.focus, 1))
                if coc * 0.45 > 0.5:
                    acc = cv2.GaussianBlur(acc, (0, 0), coc * 0.45)
            roi = img[y0:y1, x0:x1]
            roi *= 1 - acc[..., 3:4]
            roi += acc[..., :3]

    # ------------------------------------------------------------------
    # The red line (full res, own temporal samples)
    # ------------------------------------------------------------------
    def render_line(self, t):
        L = self.world.line
        mask = np.zeros((self.H, self.W), np.float32)
        # samples from camera motion and from the line's own motion
        c0, c1 = self.cam(t - SHUTTER / 2), self.cam(t + SHUTTER / 2)
        pa = L.at(t - SHUTTER / 2)[0]
        pb = L.at(t + SHUTTER / 2)[0]
        ua, _ = c0.project(pa)
        ub, _ = c1.project(pb)
        d = np.linalg.norm(ua - ub, axis=1)
        mv = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
        n = int(np.clip(math.ceil(mv / 2.5), 2, 28))
        buf = np.zeros((self.H, self.W), np.uint8)
        heads = []
        for ts in self.sub_times(t, n):
            p, w, wd, knob, gain = L.at(ts)
            if gain <= 0.002:
                continue
            c = self.cam(ts)
            uv, z = c.project(p, near=20)
            buf[:] = 0
            for i in range(len(p) - 1):
                if not (np.isfinite(uv[i]).all() and np.isfinite(uv[i + 1]).all()):
                    continue
                inten = 0.5 * (w[i] + w[i + 1])
                if inten < 0.01:
                    continue
                zz = 0.5 * (z[i] + z[i + 1])
                px = wd * c.f / zz
                thick = max(1, int(round(px)))
                val = int(np.clip(255 * inten * min(1.0, px / thick) * min(gain, 1.0), 0, 255))
                a = np.round(uv[i] * 16).astype(int)
                b = np.round(uv[i + 1] * 16).astype(int)
                if max(abs(a).max(), abs(b).max()) > 16 * 30000:
                    continue
                cv2.line(buf, tuple(int(v) for v in a), tuple(int(v) for v in b), val, thick, cv2.LINE_AA, 4)
            if knob > 0.02 and np.isfinite(uv[-1]).all():
                r = 2.2 * wd * c.f / z[-1] * knob
                if r > 0.6:
                    cv2.circle(buf, tuple(int(v) for v in np.round(uv[-1] * 16)), int(r * 16), 255, -1,
                               cv2.LINE_AA, 4)
                heads.append(uv[-1])
            mask += buf.astype(np.float32) * (max(gain, 1.0) / 255.0 / n)
        return mask

    # ------------------------------------------------------------------
    def background(self, t, cam):
        L = self.world.line
        p = L.at(t)[0]
        uv, _ = cam.project(p.mean(0, keepdims=True))
        gx, gy = uv[0] if np.isfinite(uv).all() else (self.W / 2, self.H * 0.55)
        gx = np.clip(0.5 * gx + 0.5 * self.W / 2, 0, self.W)
        gy = np.clip(0.5 * gy + 0.5 * self.H * 0.55, 0, self.H)
        R = 0.36 * self.H
        sm = 8
        xs = self.xx[::sm, ::sm]
        ys = self.yy[::sm, ::sm]
        g = np.exp(-((xs - gx) ** 2 + ((ys - gy) * 1.25) ** 2) / (2 * R * R))
        g = cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        k = 0.85 + 0.15 * smooth(lin(t, 0, 0.6))
        img = np.empty((self.H, self.W, 3), np.float32)
        img[...] = STAGE
        img[..., 0] += g * 0.042 * k
        img[..., 1] += g * 0.004 * k
        img[..., 2] += g * 0.005 * k
        return img

    def frame(self, fi):
        t = fi / FPS
        cam = self.cam(t)
        img = self.background(t, cam)
        U = self.render_universe(t, self.world.universe_gain(t))
        U = cv2.resize(U, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        img = img * (1 - U[..., 3:4]) + U[..., :3]
        self.render_heroes(img, t, cam)
        line = self.render_line(t)
        img[..., 0] += line
        # bloom from everything red-hot: the line, the icon, REC, watched bars
        red = np.clip(img[..., 0] - np.maximum(img[..., 1], img[..., 2]), 0, None)
        emis = 0.3 * red + np.minimum(line, 2.5)
        small = cv2.resize(emis, (self.W // 4, self.H // 4), interpolation=cv2.INTER_AREA)
        bloom = (0.55 * cv2.GaussianBlur(small, (0, 0), 1.6 * self.scale)
                 + 0.35 * cv2.GaussianBlur(small, (0, 0), 6 * self.scale)
                 + 0.25 * cv2.GaussianBlur(small, (0, 0), 18 * self.scale))
        bloom = cv2.resize(bloom, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        img[..., 0] += bloom * 0.8
        img[..., 1] += bloom * 0.035
        img[..., 2] += bloom * 0.03
        img *= self.vignette
        rng = np.random.default_rng(fi)
        noise = rng.normal(0, 1.0 / 255, (self.H // 2, self.W // 2)).astype(np.float32)
        img += cv2.resize(noise, (self.W, self.H), interpolation=cv2.INTER_NEAREST)[..., None]
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------
def _init(scale):
    global _W
    _W = Renderer(scale)


def _job(fi):
    return fi, _W.frame(fi).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/how_to_anything.mp4")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=DURATION)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--stills", default="")
    ap.add_argument("--crf", type=int, default=15)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"font: {A.font_name()}", file=sys.stderr)

    if args.stills:
        r = Renderer(args.scale)
        for s in args.stills.split(","):
            t = float(s)
            t0 = time.time()
            im = r.frame(int(round(t * FPS)))
            p = os.path.join(os.path.dirname(args.out) or ".", f"still_{t:05.2f}.png")
            cv2.imwrite(p, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            print(f"{p}  {time.time() - t0:.2f}s", file=sys.stderr)
        return

    W, H = int(WIDTH * args.scale), int(HEIGHT * args.scale)
    f0 = int(round(args.start * FPS))
    f1 = int(round(args.end * FPS))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-tune", "film",
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-color_range", "tv", "-movflags", "+faststart", args.out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.scale,)) as pool:
        for k, (fi, data) in enumerate(pool.imap(_job, range(f0, f1), chunksize=1)):
            enc.stdin.write(data)
            if k % 30 == 0:
                el = time.time() - t0
                print(f"frame {fi}/{f1}  t={fi / FPS:5.2f}s  {el:6.1f}s elapsed", file=sys.stderr, flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"wrote {args.out} ({f1 - f0} frames, {time.time() - t0:.0f}s; main {MAIN_DUR}s + end card)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
