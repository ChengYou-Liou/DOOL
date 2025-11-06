import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load numerical data
u_df = pd.read_csv("result_1d_T1_dt0.001/n_solution_test.csv",
                   header=None, delim_whitespace=True).apply(pd.to_numeric, errors="coerce")
j_df = pd.read_csv("result_1d_T1_dt0.001/j_solution_test.csv",
                   header=None, delim_whitespace=True).apply(pd.to_numeric, errors="coerce")

u_num = u_df.to_numpy()  # shape: (n_space, n_time)
j_num = j_df.to_numpy()

n_space, n_time = u_num.shape
print(f"Grid: space={n_space}, time={n_time}")

# Physical grids
x_min, x_max = -np.pi, np.pi
t_min, t_max = 0.0, 1.0
x = np.linspace(x_min, x_max, n_space)
t = np.linspace(t_min, t_max, n_time)

# Mesh with rows=x, cols=t to match u_num[i,j] = u(x_i, t_j)
X, T = np.meshgrid(x, t, indexing="ij")

# Analytical solutions
u_anal = np.exp(-T) * np.sin(X) + 2.0
j_anal = -np.exp(-T) * np.cos(X)          # j = -u_x with diffusivity = 1

# Errors & RMSE
u_err = u_num - u_anal
j_err = j_num - j_anal
u_rmse = float(np.sqrt(np.mean(u_err**2)))
j_rmse = float(np.sqrt(np.mean(j_err**2)))
print(f"RMSE(u): {u_rmse:.6e},  RMSE(j): {j_rmse:.6e}")

# Plots: u and j side-by-side
extent = [t_min, t_max, x_min, x_max]

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)

# u: numerical / analytical / error
im0 = axes[0,0].imshow(u_num, origin="lower", aspect="auto", cmap="seismic", extent=extent)
axes[0,0].set_title("Numerical u(x,t)"); axes[0,0].set_ylabel("x")
plt.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)

im1 = axes[0,1].imshow(u_anal, origin="lower", aspect="auto", cmap="seismic", extent=extent)
axes[0,1].set_title("Analytical u(x,t)")
plt.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)

im2 = axes[0,2].imshow(u_err, origin="lower", aspect="auto", cmap="seismic", extent=extent)
axes[0,2].set_title(f"u error\nRMSE={u_rmse:.2e}")
plt.colorbar(im2, ax=axes[0,2], fraction=0.046, pad=0.04)

# j: numerical / analytical / error
im3 = axes[1,0].imshow(j_num, origin="lower", aspect="auto", cmap="seismic", extent=extent)
axes[1,0].set_title("Numerical j(x,t)"); axes[1,0].set_xlabel("t"); axes[1,0].set_ylabel("x")
plt.colorbar(im3, ax=axes[1,0], fraction=0.046, pad=0.04)

im4 = axes[1,1].imshow(j_anal, origin="lower", aspect="auto", cmap="seismic", extent=extent)
axes[1,1].set_title("Analytical j(x,t)"); axes[1,1].set_xlabel("t")
plt.colorbar(im4, ax=axes[1,1], fraction=0.046, pad=0.04)

im5 = axes[1,2].imshow(j_err, origin="lower", aspect="auto", cmap="seismic", extent=extent)
axes[1,2].set_title(f"j error\nRMSE={j_rmse:.2e}"); axes[1,2].set_xlabel("t")
plt.colorbar(im5, ax=axes[1,2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("u_j_compare_with_analytical.png", dpi=200)
#plt.show()

# Optional: conservation residual check
# Compute finite-difference residual R = u_t + j_x (should be ~0)
# central differences in interior, one-sided at boundaries

# time derivative u_t
dt = (t_max - t_min) / (n_time - 1) if n_time > 1 else 1.0
dx = (x_max - x_min) / (n_space - 1) if n_space > 1 else 1.0

u_t = np.zeros_like(u_num)
if n_time > 1:
    u_t[:,1:-1] = (u_num[:,2:] - u_num[:,:-2]) / (2*dt)
    u_t[:,0]    = (u_num[:,1] - u_num[:,0]) / dt
    u_t[:,-1]   = (u_num[:,-1] - u_num[:,-2]) / dt

# space derivative j_x
j_x = np.zeros_like(j_num)
if n_space > 1:
    j_x[1:-1,:] = (j_num[2:,:] - j_num[:-2,:]) / (2*dx)
    j_x[0,:]    = (j_num[1,:] - j_num[0,:]) / dx
    j_x[-1,:]   = (j_num[-1,:] - j_num[-2,:]) / dx

R = u_t + j_x
res_norm = float(np.linalg.norm(R) / np.sqrt(R.size))
print(f"Conservation residual ||u_t + j_x||_RMS = {res_norm:.6e}")

# Optional: visualize residual
plt.figure(figsize=(6,4))
im = plt.imshow(R, origin="lower", aspect="auto", cmap="seismic", extent=extent, vmin=1e-4, vmax=1e-3)
plt.title(f"Conservation residual u_t + j_x (RMS={res_norm:.2e})")
plt.xlabel("t"); plt.ylabel("x")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("conservation_residual.png", dpi=200)
#plt.show()
