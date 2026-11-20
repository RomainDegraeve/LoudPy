"""
Animations of fields (matplotlib, loudpy style): written to .mp4 (ffmpeg, from imageio-ffmpeg) or .gif.

    lp.plot.animate(trans, "u.mp4", name="u")                  a transient: one frame per (sub-sampled) step
    lp.plot.animate(res.field("u", f=1000), "u_1k.gif")        a harmonic field over one period
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from loudpy.plot import style


def _writer(path, fps):
    from matplotlib import animation
    if Path(path).suffix.lower() == ".gif":
        return animation.PillowWriter(fps=fps)
    try:
        import imageio_ffmpeg
        import matplotlib as mpl
        mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return animation.FFMpegWriter(fps=fps, bitrate=4000)


def _frames(source, name, start, duration, fps, n_periods):
    from loudpy.results.data import Field
    if isinstance(source, Field):
        phases = np.linspace(0, 2 * np.pi * n_periods, max(2, int(duration * fps)), endpoint=False)
        return [Field(source.name, source.mesh, np.real(source.values * np.exp(1j * p)), source.coordinate,
                      source.coord, source.extra, source.quantity, {**source.meta, "phase": p}) for p in phases]
    t = source.values
    i0 = int(np.searchsorted(t, start)) if start is not None else 0
    stride = max(1, (len(t) - i0) // max(1, int(fps * duration)))
    return [source.field(name, index=i) for i in range(i0, len(t), stride)]


def animate(source, path, *, name: str = "u", deformed: bool = True, scale: float = 0.05, start: float | None = None,
            duration: float = 8.0, fps: int = 30, n_periods: int = 1, **kwargs):
    """
    Animate a field and write it to `path` (.mp4 or .gif); returns the path.

    source    : TimeResults (field `name`, one frame per sub-sampled step from `start` [s]) or a harmonic
                Field (its real part over n_periods periods)
    deformed  : vector fields drawn on the deformed structure, one deformation scale for all frames
                (peak = `scale` × mesh size); colour = |field| on a fixed range
    duration, fps : length [s] and frame rate of the animation
    kwargs    : passed to lp.plot.field (xlim, ylim, mirror, cmap …)
    """
    import matplotlib.pyplot as plt
    from loudpy.plot.fields import field as draw
    frames = _frames(source, name, start, duration, fps, n_periods)
    if not frames:
        raise ValueError("nothing to animate")
    vec = frames[0].is_vector
    peak = max(float(np.max(np.linalg.norm(np.real(fr.values), axis=1) if vec else np.abs(np.real(fr.values))))
               for fr in frames) or 1.0
    factor = scale * float(np.ptp(frames[0].mesh.coords, axis=0).max()) / peak
    fig, ax = style.figure("field")
    writer = _writer(path, fps)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cbar = None
    with writer.saving(fig, str(path), dpi=150):
        for fr in frames:
            ax.clear()
            if vec:
                mag = fr._scalar(np.linalg.norm(np.real(fr.values), axis=1), fr.quantity.part("abs"))
                draw(mag, ax=ax, deformed=fr if deformed else None, factor=factor, vmin=0, vmax=peak,
                     colorbar=cbar is None, **kwargs)
            else:
                draw(fr, show="real", ax=ax, vmin=-peak, vmax=peak, colorbar=cbar is None, **kwargs)
            cbar = True
            writer.grab_frame()
    plt.close(fig)
    return Path(path)
