import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
import numpy as np
import argparse
import imageio.v3 as iio
import io

# external GIF generators
from generate_linecurve_gifs import (
    generate_residual_gif,
    generate_rho_gif,
    generate_j_gif
)


parser = argparse.ArgumentParser(description='hyper parameters')
parser.add_argument('--d', type=int, default=4, help='depth')
parser.add_argument('--n', type=int, default=70, help='width')
parser.add_argument('--k', type=int, default=120, help='width of the output layer')
parser.add_argument('--nx', type=int, default=128, help='Sampling')

args = parser.parse_args()

# ------------------------------------------
# Load numerical data
# ------------------------------------------
u_df = pd.read_csv(
    "result_1d_T1_dt0.001/n_solution_test.csv",
    header=None,
    sep=r"\s+"
).apply(pd.to_numeric, errors="coerce")

j_df = pd.read_csv(
    "result_1d_T1_dt0.001/j_solution_test.csv",
    header=None,
    sep=r"\s+"
).apply(pd.to_numeric, errors="coerce")


# Load training log
results_df = pd.read_csv(
    f"DeepONet_training_d{args.d}_n{args.n}_nx{args.nx}_k{args.k}/Phase_training_data.csv",
    delimiter=",", engine="python", header=0, index_col=None
)

u_num = u_df.to_numpy()  # shape: (n_space, n_time)
j_num = j_df.to_numpy()

n_space, n_time = u_num.shape
print(f"Grid: space={n_space}, time={n_time}")

# ------------------------------------------
# Physical grids
# ------------------------------------------
x_min, x_max = -np.pi, np.pi
t_min, t_max = 0.0, 1.0
x = np.linspace(x_min, x_max, n_space)
t = np.linspace(t_min, t_max, n_time)

X, T = np.meshgrid(x, t, indexing="ij")

# ------------------------------------------
# Analytical solutions
# ------------------------------------------
u_anal = np.exp(-T) * np.sin(X) + 2.0
j_anal = -np.exp(-T) * np.cos(X)

# ------------------------------------------
# Errors & MSE
# ------------------------------------------
u_err = u_num - u_anal
j_err = j_num - j_anal
u_mse = float(np.mean(u_err**2))
j_mse = float(np.mean(j_err**2))

print(f"MSE(u/ρ): {u_mse:.6e},  MSE(j): {j_mse:.6e}")

# ------------------------------------------
# Heatmap plots (u, j, errors)
# ------------------------------------------
extent = [t_min, t_max, x_min, x_max]
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)

# u heatmaps
im0 = axes[0,0].imshow(u_num, origin="lower", aspect="auto", cmap="seismic", extent=extent)
axes[0,0].set_title("Numerical ρ(x,t)"); axes[0,0].set_ylabel("x")
plt.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)

im1 = axes[0,1].imshow(u_anal, origin="lower", aspect="auto", cmap="seismic", extent=extent)
axes[0,1].set_title("Analytical ρ(x,t)")
plt.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)

im2 = axes[0,2].imshow(u_err, origin="lower", aspect="auto", cmap="seismic", extent=extent)
axes[0,2].set_title(f"ρ error\nMSE={u_mse:.2e}")
plt.colorbar(im2, ax=axes[0,2], fraction=0.046, pad=0.04)

# j heatmaps
im3 = axes[1,0].imshow(j_num, origin="lower", aspect="auto", cmap="seismic", extent=extent)
axes[1,0].set_title("Numerical j(x,t)"); axes[1,0].set_xlabel("t"); axes[1,0].set_ylabel("x")
plt.colorbar(im3, ax=axes[1,0], fraction=0.046, pad=0.04)

im4 = axes[1,1].imshow(j_anal, origin="lower", aspect="auto", cmap="seismic", extent=extent)
axes[1,1].set_title("Analytical j(x,t)"); axes[1,1].set_xlabel("t")
plt.colorbar(im4, ax=axes[1,1], fraction=0.046, pad=0.04)

im5 = axes[1,2].imshow(j_err, origin="lower", aspect="auto", cmap="seismic", extent=extent)
axes[1,2].set_title(f"j error\nMSE={j_mse:.2e}"); axes[1,2].set_xlabel("t")
plt.colorbar(im5, ax=axes[1,2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("u_j_original.png", dpi=200)

# ------------------------------------------
# Compute conservation residual R = u_t + j_x
# ------------------------------------------
dt = (t_max - t_min) / (n_time - 1)
dx = (x_max - x_min) / (n_space - 1)

u_t = np.zeros_like(u_num)
u_t[:,1:-1] = (u_num[:,2:] - u_num[:,:-2]) / (2*dt)
u_t[:,0]    = (u_num[:,1] - u_num[:,0]) / dt
u_t[:,-1]   = (u_num[:,-1] - u_num[:,-2]) / dt

j_x = np.zeros_like(j_num)
j_x[1:-1,:] = (j_num[2:,:] - j_num[:-2,:]) / (2*dx)
j_x[0,:]    = (j_num[1,:] - j_num[0,:]) / dx
j_x[-1,:]   = (j_num[-1,:] - j_num[-2,:]) / dx

R = u_t + j_x

# ----------------------------------------------------
# Compute per-time MSE for rho, j, and residual
# ----------------------------------------------------

rho_mse_t = np.mean((u_num - u_anal)**2, axis=0)   # shape: (n_time,)
j_mse_t   = np.mean((j_num - j_anal)**2, axis=0)
res_mse_t = np.mean(R**2, axis=0)

# ----------------------------------------------------
# Compute max MSE over time
# ----------------------------------------------------

rho_mse_max = float(np.max(rho_mse_t))
rho_mse_argmax = int(np.argmax(rho_mse_t))

j_mse_max = float(np.max(j_mse_t))
j_mse_argmax = int(np.argmax(j_mse_t))

res_mse_max = float(np.max(res_mse_t))
res_mse_argmax = int(np.argmax(res_mse_t))

# ----------------------------------------------------
# Print results to terminal
# ----------------------------------------------------
print("\n========== MAX MSE OVER TIME ==========")

print(f"Max ρ MSE  = {rho_mse_max:.6e}   at t_idx = {rho_mse_argmax},   t = {t[rho_mse_argmax]:.4f}")
print(f"Max j MSE  = {j_mse_max:.6e}     at t_idx = {j_mse_argmax},     t = {t[j_mse_argmax]:.4f}")
print(f"Max R MSE  = {res_mse_max:.6e}   at t_idx = {res_mse_argmax},   t = {t[res_mse_argmax]:.4f}")

print("========================================\n")


res_mse = float(np.mean(R**2))
print(f"Conservation MSE(u_t + j_x) = {res_mse:.6e}")


# plt.figure(figsize=(6,4))
# im = plt.imshow(R**2,
#                 origin="lower",
#                 aspect="auto",
#                 cmap="seismic",
#                 extent=extent,
#                 vmin=0.0,
#                 vmax=0.001)
# plt.title(f"Conservation residual MSE(u_t + j_x) = {res_mse:.2e}")
# plt.xlabel("t"); plt.ylabel("x")
# plt.colorbar(im, fraction=0.046, pad=0.04)
# plt.tight_layout()
# plt.savefig("conservation_residual_original.png", dpi=200)

# ------------------------------------------
# Loss curve
# ------------------------------------------
iters = results_df["Iter"]
losses = results_df["Loss"]

plt.figure(figsize=(8,5))
plt.plot(iters, losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("DeepONet Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    f"DeepONet_training_d{args.d}_n{args.n}_nx{args.nx}_k{args.k}/loss_curve_original.png",
    dpi=200
)

print(results_df.head())
print(results_df.dtypes)

# ------------------------------------------
# Generate GIFs (now using functions)
# ------------------------------------------
generate_residual_gif(R, x, t, filename="residual_original.gif", fps=30)
generate_rho_gif(u_num, u_anal, x, t, filename="rho_DOOL_original.gif", fps=30)
generate_j_gif(j_num, j_anal, x, t, filename="j_DOOL_original.gif", fps=30)
