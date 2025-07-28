import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# read .mat file
data = loadmat('heat_ground_truth.mat')
X = data['X_flat'].flatten()
Y = data['Y_flat'].flatten()
T = data['T_flat'].flatten()
U = data['U_flat'].flatten()

# print original shape
for key in data:
    value = data[key]
    if isinstance(value, np.ndarray):
        print(f"{key}: shape = {value.shape}")
    else:
        print(f"{key}: type = {type(value)} (no shape)")

# print flattened shape
for key, value in {"X": X, "Y": Y, "T": T, "U": U}.items():
    print(f"{key}_flat flattened shape: {value.shape}")

# choose time slice [0.0, 3.0]
target_t = 0

# find closest time step
closest_index = np.argmin(np.abs(np.unique(T) - target_t))
closest_t = np.unique(T)[closest_index]
idx = np.where(np.isclose(T, closest_t))[0]
print(f"You requested t = {target_t}")
print(f"Closest match t = {closest_t}")
print(f"Closest time step index = {closest_index}")

# 64 * 64 mesh
Nx = 64
Ny = 64
x_slice = X[idx]
y_slice = Y[idx]
u_slice = U[idx]

# reshape to 2D mesh
X_grid = x_slice.reshape(Nx, Ny)
Y_grid = y_slice.reshape(Nx, Ny)
U_grid = u_slice.reshape(Nx, Ny)

# graphing 3D surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_grid, Y_grid, U_grid, cmap='viridis', edgecolor='none')
ax.set_title(f"Analytical Solution at t = {target_t}")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x, y, t)', labelpad=20)
fig.colorbar(surf, ax=ax, shrink=0.6, label='u')

# transfer z-axis to left
ax.zaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=5)
ax.zaxis._axinfo['juggled'] = (1, 1, 0)

plt.tight_layout()
plt.savefig("plot.png", dpi=300)
plt.show()
