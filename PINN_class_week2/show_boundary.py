import os
import matplotlib
matplotlib.use("Agg")  # 非互動模式

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.geometry.primitives_3d import Box

# ===== 產生 geometry 樣本 =====
line = Line1D(0, 1)
rect = Rectangle((0, 0), (1, 1))
box = Box((0, 0, 0), (1, 1, 1))

samples_1d = line.sample_boundary(2)
samples_2d = rect.sample_boundary(200)
samples_3d = box.sample_boundary(5000)

# ===== 建立輸出資料夾 =====
output_dir = "show_boundary"
os.makedirs(output_dir, exist_ok=True)

# ===== 繪圖並儲存圖像 =====
fig = plt.figure(figsize=(18, 5))

# 1D plot
ax1 = fig.add_subplot(131)
ax1.scatter(samples_1d["x"], [0]*len(samples_1d["x"]), c="red", s=100)
ax1.set_title("1D Boundary (Line)")
ax1.set_xlabel("x")
ax1.set_yticks([])
ax1.grid(True)

# 2D plot
ax2 = fig.add_subplot(132)
ax2.scatter(samples_2d["x"], samples_2d["y"], c="blue", s=5)
ax2.set_title("2D Boundary (Rectangle)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_aspect("equal")
ax2.grid(True)

# 3D plot
ax3 = fig.add_subplot(133, projection="3d")
ax3.scatter(samples_3d["x"], samples_3d["y"], samples_3d["z"], c="green", s=3)
ax3.set_title("3D Boundary (Box)")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("z")

plt.tight_layout()
output_path = os.path.join(output_dir, "boundary_plot.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")