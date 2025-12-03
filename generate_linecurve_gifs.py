import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
import imageio.v3 as iio
import io


# ------------------------------------------
# Residual GIF (log-scale)
# ------------------------------------------
def generate_residual_gif(R, x, t, filename="residual_original.gif", fps=30):

    eps = 1e-15
    R_abs = np.abs(R) + eps

    global_min = R_abs.min() * 0.8
    global_max = R_abs.max() * 1.2

    n_time = R.shape[1]
    res_t_mse = np.mean(R**2, axis=0)
    time_indices = range(0, n_time, 4)

    frames = []

    for j in time_indices:
        fig, ax = plt.subplots(figsize=(6,4))

        ax.semilogy(x, R_abs[:, j], label="|Residual| (log scale)", linewidth=2)

        mse_t = res_t_mse[j]
        ax.set_title(f"Residual |R(x,t)|:  t={t[j]:.3f},  MSE={mse_t:.3e}")

        ax.set_xlabel("x")
        ax.set_ylabel("|R| (log scale)")
        ax.grid(True, which="both")
        ax.legend(loc="upper right")

        ax.set_ylim(global_min, global_max)
        ax.yaxis.set_major_formatter(LogFormatterMathtext())

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(iio.imread(buf))

    iio.imwrite(filename, frames, fps=fps)
    print(f"[OK] Residual GIF saved → {filename}")


# ------------------------------------------
# ρ(x,t) GIF  (u → rho)
# ------------------------------------------
def generate_rho_gif(u_num, u_anal, x, t, filename="rho_DOOL_original.gif", fps=30):

    y_min = min(u_num.min(), u_anal.min())
    y_max = max(u_num.max(), u_anal.max())

    n_time = u_num.shape[1]
    time_indices = range(0, n_time, 4)

    frames = []

    for j in time_indices:

        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(x, u_num[:, j], label=r"$\rho$ (DOOL)", linewidth=2)
        ax.plot(x, u_anal[:, j], '--', label=r"$\rho$ (Analytical)", linewidth=2)

        mse_t = np.mean((u_num[:, j] - u_anal[:, j])**2)

        ax.set_title(rf"$\rho(x,t)$:  t={t[j]:.3f},  MSE={mse_t:.3e}")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\rho(x,t)$")
        ax.grid(True)
        ax.legend(loc="upper right")

        ax.set_ylim(y_min, y_max)

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(iio.imread(buf))

    iio.imwrite(filename, frames, fps=fps)
    print(f"[OK] rho(x,t) GIF saved → {filename}")


# ------------------------------------------
# j(x,t) GIF  (DOOL vs analytical)
# ------------------------------------------
def generate_j_gif(j_num, j_anal, x, t, filename="j_DOOL_original.gif", fps=30):

    y_min = min(j_num.min(), j_anal.min())
    y_max = max(j_num.max(), j_anal.max())

    n_time = j_num.shape[1]
    time_indices = range(0, n_time, 4)

    frames = []

    for j in time_indices:

        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(x, j_num[:, j], label=r"$j$ (DOOL)", linewidth=2)
        ax.plot(x, j_anal[:, j], '--', label=r"$j$ (Analytical)", linewidth=2)

        mse_t = np.mean((j_num[:, j] - j_anal[:, j])**2)

        ax.set_title(rf"$j(x,t)$:  t={t[j]:.3f},  MSE={mse_t:.3e}")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$j(x,t)$")
        ax.grid(True)
        ax.legend(loc="upper right")

        ax.set_ylim(y_min, y_max)

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(iio.imread(buf))

    iio.imwrite(filename, frames, fps=fps)
    print(f"[OK] j(x,t) GIF saved → {filename}")
