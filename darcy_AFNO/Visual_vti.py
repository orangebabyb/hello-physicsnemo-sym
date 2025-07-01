import pyvista as pv

# 讀檔
#grid = pv.read("outputs/darcy_AFNO/validators/test.vti")
grid = pv.read("outputs/custom_darcy_AFNO/validators/test_0.vti")

# 印出完整資訊（含結構、陣列名稱、資料型態等）
print(grid)

# 印出所有欄位（也就是每個變數/通道）
print("\n[欄位名稱] Available arrays:", grid.point_data.keys())

# 印出特定欄位的統計資訊（例如 true_sol)
print("【所有欄位名稱與資料型態】")
for name in grid.point_data.keys():
    arr = grid[name]
    print(f"{name}:{arr}")
    #print(f"- {name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}, mean={arr.mean()}")