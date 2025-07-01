import pyvista as pv

# read vti file
grid = pv.read("outputs/custom_darcy_AFNO/validators/test_0.vti")

# print dict struct
print(grid)

# print all column
print("\n[欄位名稱] Available arrays:", grid.point_data.keys())

print("[所有欄位名稱與資料型態]")
for name in grid.point_data.keys():
    arr = grid[name]
    print(f"{name}:{arr}")
    #print(f"- {name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}, mean={arr.mean()}")
