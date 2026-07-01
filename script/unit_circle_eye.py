"""Unit circle drawn as a stylized eye.

Renders the standard trig unit circle — special angles in degrees & radians
with their (cos theta, sin theta) coordinates around the rim — but the disk
itself is painted as an iris: radial fibers, a layered green/teal/blue gradient,
a dark pupil and a catch-light. Run it to write fig/unit_circle_eye.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

# ----------------------------------------------------------------------------
# Special angles: (degrees, radian-label, (x, y) exact-form coordinate label)
# ----------------------------------------------------------------------------
SQ2 = r"\frac{\sqrt{2}}{2}"
SQ3 = r"\frac{\sqrt{3}}{2}"
HALF = r"\frac{1}{2}"

ANGLES = [
    (0,   r"0",            "(1, 0)"),
    (30,  r"\frac{\pi}{6}", rf"$\left({SQ3},\,{HALF}\right)$"),
    (45,  r"\frac{\pi}{4}", rf"$\left({SQ2},\,{SQ2}\right)$"),
    (60,  r"\frac{\pi}{3}", rf"$\left({HALF},\,{SQ3}\right)$"),
    (90,  r"\frac{\pi}{2}", "(0, 1)"),
    (120, r"\frac{2\pi}{3}", rf"$\left(-{HALF},\,{SQ3}\right)$"),
    (135, r"\frac{3\pi}{4}", rf"$\left(-{SQ2},\,{SQ2}\right)$"),
    (150, r"\frac{5\pi}{6}", rf"$\left(-{SQ3},\,{HALF}\right)$"),
    (180, r"\pi",           "(-1, 0)"),
    (210, r"\frac{7\pi}{6}", rf"$\left(-{SQ3},\,-{HALF}\right)$"),
    (225, r"\frac{5\pi}{4}", rf"$\left(-{SQ2},\,-{SQ2}\right)$"),
    (240, r"\frac{4\pi}{3}", rf"$\left(-{HALF},\,-{SQ3}\right)$"),
    (270, r"\frac{3\pi}{2}", "(0, -1)"),
    (300, r"\frac{5\pi}{3}", rf"$\left({HALF},\,-{SQ3}\right)$"),
    (315, r"\frac{7\pi}{4}", rf"$\left({SQ2},\,-{SQ2}\right)$"),
    (330, r"\frac{11\pi}{6}", rf"$\left({SQ3},\,-{HALF}\right)$"),
]


def iris_colormap():
    """Hand-tuned teal -> green -> deep-blue radial palette for the iris."""
    stops = ["#0c3b46", "#117a8b", "#2bb3a3", "#5fd6a0",
             "#2f8f6f", "#1f6f8b", "#16557e", "#0d2f5c"]
    rgb = np.array([to_rgb(c) for c in stops])
    t = np.linspace(0, 1, len(stops))
    return t, rgb


def sample_iris(r, t, rgb):
    """Interpolate the iris palette at normalized radii r in [0, 1]."""
    return np.stack([np.interp(r, t, rgb[:, i]) for i in range(3)], axis=-1)


def draw_eye(ax, rng):
    """Paint the iris: a smooth radial gradient with faint fibers, pupil,
    catch-light."""
    t, rgb = iris_colormap()
    pupil_r, iris_r = 0.30, 1.0

    # Smooth radial gradient rendered as a fine pixel image. Each pixel's
    # color is interpolated from the iris palette by its distance to center,
    # so the color transitions are continuous with no visible banding.
    res = 1200
    grid = np.linspace(-iris_r, iris_r, res)
    gx, gy = np.meshgrid(grid, grid)
    rr = np.sqrt(gx ** 2 + gy ** 2)
    frac = np.clip((rr - pupil_r) / (iris_r - pupil_r), 0, 1)
    img = sample_iris(frac, t, rgb)               # (res, res, 3)
    alpha = np.where(rr <= iris_r, 1.0, 0.0)[..., None]
    img = np.concatenate([img, alpha], axis=-1)   # mask outside the disk
    ax.imshow(img, extent=[-iris_r, iris_r, -iris_r, iris_r],
              origin="lower", zorder=1, interpolation="bilinear")

    # Whisper-faint radial fibers: just enough to suggest iris structure
    # without breaking the smoothness of the color field.
    n_fibers = 300
    segs, colors, widths = [], [], []
    for k in range(n_fibers):
        ang = 2 * np.pi * k / n_fibers + rng.uniform(-0.01, 0.01)
        r0 = pupil_r + rng.uniform(0.0, 0.05)
        r1 = iris_r * rng.uniform(0.85, 0.99)
        n_pts = 7
        rs = np.linspace(r0, r1, n_pts)
        wob = np.cumsum(rng.uniform(-0.008, 0.008, n_pts))
        xs = rs * np.cos(ang + wob)
        ys = rs * np.sin(ang + wob)
        pts = np.column_stack([xs, ys])
        for j in range(n_pts - 1):
            f = 1 - (rs[j] - pupil_r) / (iris_r - pupil_r)
            base = sample_iris(np.clip(f, 0, 1), t, rgb)
            col = np.clip(base + rng.uniform(-0.03, 0.05), 0, 1)
            segs.append(pts[j:j + 2])
            colors.append((*col, rng.uniform(0.05, 0.16)))
            widths.append(rng.uniform(0.3, 0.9))
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths,
                                     zorder=2, capstyle="round"))

    # Limbal ring: dark rim that frames the iris.
    ax.add_patch(Circle((0, 0), iris_r, fill=False, ec="#06222b",
                        lw=6, zorder=3, alpha=0.9))
    ax.add_patch(Circle((0, 0), iris_r * 0.995, fill=False, ec="#0a3340",
                        lw=2, zorder=3, alpha=0.7))

    # Pupil with a faint glow ring.
    ax.add_patch(Circle((0, 0), pupil_r * 1.12, color="#04161c",
                        lw=0, zorder=3, alpha=0.5))
    ax.add_patch(Circle((0, 0), pupil_r, color="#020a0d", lw=0, zorder=4))
    ax.add_patch(Circle((0, 0), pupil_r, fill=False, ec="#16557e",
                        lw=1.2, zorder=5, alpha=0.5))

    # Catch-light: the spark that makes it read as an eye.
    ax.add_patch(Circle((-0.11, 0.12), 0.085, color="#f3fbff",
                        lw=0, zorder=6, alpha=0.95))
    ax.add_patch(Circle((0.07, -0.07), 0.03, color="#dff4ff",
                        lw=0, zorder=6, alpha=0.6))


def main():
    rng = np.random.default_rng(7)  # fixed seed -> reproducible texture

    fig, ax = plt.subplots(figsize=(13, 13), dpi=150)
    fig.patch.set_facecolor("#f4f1ea")   # warm paper
    ax.set_facecolor("#f4f1ea")

    draw_eye(ax, rng)

    tick_in, tick_out = 1.0, 1.06
    radian_r = 0.78          # radian/degree labels ride inside the rim
    coord_r = 1.12           # coordinate labels hug the rim

    for deg, rad, coord in ANGLES:
        a = np.radians(deg)
        ca, sa = np.cos(a), np.sin(a)

        # Tick mark + dot on the circle at each special angle.
        ax.plot([tick_in * ca, tick_out * ca], [tick_in * sa, tick_out * sa],
                color="#1c1c1c", lw=1.4, zorder=7)
        ax.plot(ca, sa, "o", ms=6, color="#0b2a33",
                mec="#f4f1ea", mew=1.0, zorder=8)

        # Degree = radian label, rotated to sit along the spoke, inside the rim.
        rot = deg if deg <= 90 or deg >= 270 else deg - 180
        ax.text(radian_r * ca, radian_r * sa,
                rf"${deg}^\circ\!=\!{rad}$",
                color="#0a1f25", fontsize=11, ha="center", va="center",
                rotation=rot, rotation_mode="anchor", zorder=7,
                fontweight="medium")

        # Coordinate label outside, alignment follows the quadrant.
        ha = "center" if abs(ca) < 1e-9 else ("left" if ca > 0 else "right")
        va = "center" if abs(sa) < 1e-9 else ("bottom" if sa > 0 else "top")
        ax.text(coord_r * ca, coord_r * sa, coord, color="#15303a",
                fontsize=17, ha=ha, va=va, zorder=7)

    # (cos theta, sin theta) caption, upper right, like the reference.
    ax.text(1.12, 1.38, r"$(\cos\theta,\ \sin\theta)$", fontsize=16,
            color="#15303a", ha="center", va="center", style="italic")
    ax.text(0, -1.48, "the unit circle", fontsize=16, color="#26474f",
            ha="center", va="center", style="italic", alpha=0.8)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.55, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "fig")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "unit_circle_eye.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
