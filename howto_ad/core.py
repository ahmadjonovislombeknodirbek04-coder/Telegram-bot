"""Math core: easing, monotone splines, and a real pinhole perspective camera."""
import numpy as np

# ----------------------------------------------------------------------------
# Easing
# ----------------------------------------------------------------------------

def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def lin(t, a, b):
    """0..1 ramp of t over [a, b]."""
    if b <= a:
        return 1.0 if t >= b else 0.0
    return float(clamp01((t - a) / (b - a)))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp01(x)
    return x * x * x * (x * (6 * x - 15) + 10)


def ease_out_cubic(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in_out_cubic(x):
    x = clamp01(x)
    return np.where(x < 0.5, 4 * x ** 3, 1 - (-2 * x + 2) ** 3 / 2) if np.ndim(x) else (
        4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2)


def ease_out_expo(x):
    x = float(clamp01(x))
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def ease_out_back(x, s=1.70158):
    x = float(clamp01(x)) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def window(t, a, b, fade_in, fade_out):
    """1 inside [a, b] with smooth fades of the given lengths outside it."""
    return smooth(lin(t, a - fade_in, a)) * (1 - smooth(lin(t, b, b + fade_out)))


# ----------------------------------------------------------------------------
# Monotone cubic (PCHIP) interpolation – no overshoot, so the camera never
# reverses or stalls between keys that move in one direction.
# ----------------------------------------------------------------------------

class Pchip:
    def __init__(self, ts, ys):
        self.t = np.asarray(ts, float)
        self.y = np.asarray(ys, float)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        h = np.diff(self.t)[:, None]
        d = np.diff(self.y, axis=0) / h
        n = len(self.t)
        m = np.zeros_like(self.y)
        for i in range(1, n - 1):
            a, b = d[i - 1], d[i]
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            same = (a * b) > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                hm = (w1 + w2) / (w1 / a + w2 / b)
            m[i] = np.where(same, hm, 0.0)
        m[0] = self._end(h[0], h[1] if n > 2 else h[0], d[0], d[1] if n > 2 else d[0])
        m[-1] = self._end(h[-1], h[-2] if n > 2 else h[-1], d[-1], d[-2] if n > 2 else d[-1])
        self.m = m

    @staticmethod
    def _end(h0, h1, d0, d1):
        m = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        m = np.where(np.sign(m) != np.sign(d0), 0.0, m)
        m = np.where((np.sign(d0) != np.sign(d1)) & (np.abs(m) > np.abs(3 * d0)), 3 * d0, m)
        return m

    def __call__(self, t):
        t = float(np.clip(t, self.t[0], self.t[-1]))
        i = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        h = self.t[i + 1] - self.t[i]
        s = (t - self.t[i]) / h
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        return (h00 * self.y[i] + h10 * h * self.m[i] + h01 * self.y[i + 1] + h11 * h * self.m[i + 1])


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------

def look_rotation(pos, target, roll_deg):
    """World->camera rotation. World and camera are both x-right, y-down, z-forward."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    down = np.array([0.0, 1.0, 0.0])
    right = np.cross(down, fwd)
    right /= np.linalg.norm(right)
    up_d = np.cross(fwd, right)
    r = np.radians(roll_deg)
    c, s = np.cos(r), np.sin(r)
    xr = c * right + s * up_d
    yr = -s * right + c * up_d
    return np.stack([xr, yr, fwd])


class Camera:
    def __init__(self, pos, R, f, w, h, focus):
        self.pos = np.asarray(pos, float)
        self.R = R
        self.f = f
        self.w, self.h = w, h
        self.cx, self.cy = w / 2.0, h / 2.0
        self.focus = focus

    def to_cam(self, P):
        P = np.asarray(P, float)
        return (P - self.pos) @ self.R.T

    def project(self, P, near=1.0):
        """Returns (uv, z). uv is NaN where the point is behind the near plane."""
        pc = self.to_cam(P)
        z = pc[..., 2]
        zs = np.where(z > near, z, np.nan)
        u = self.f * pc[..., 0] / zs + self.cx
        v = self.f * pc[..., 1] / zs + self.cy
        return np.stack([u, v], -1), z

    def scaled(self, k):
        return Camera(self.pos, self.R, self.f * k, self.w * k, self.h * k, self.focus)


class CameraRig:
    """Keyframed camera: position, look-target and roll, each through PCHIP."""

    def __init__(self, keys, f, w, h):
        ts = [k[0] for k in keys]
        self.pos = Pchip(ts, [k[1] for k in keys])
        self.tgt = Pchip(ts, [k[2] for k in keys])
        self.roll = Pchip(ts, [[k[3]] for k in keys])
        self.f, self.w, self.h = f, w, h

    def at(self, t):
        p = self.pos(t)
        g = self.tgt(t)
        R = look_rotation(p, g, float(self.roll(t)[0]))
        focus = float(np.dot(g - p, R[2]))
        return Camera(p, R, self.f, self.w, self.h, focus)


def plane_corners(center, w, h, rot=(0.0, 0.0, 0.0)):
    """Corners (TL, TR, BR, BL) of a w x h plane facing -z, rotated by yaw/pitch/roll (deg)."""
    yaw, pitch, roll = np.radians(rot)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    M = Ry @ Rx @ Rz
    local = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
    return np.asarray(center, float) + local @ M.T, M
