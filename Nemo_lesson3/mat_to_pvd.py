import os
import numpy as np
from scipy.io import loadmat
import vtk
from vtk.util.numpy_support import numpy_to_vtk

# Setup parameters
mat_file = "heat_ground_truth.mat"
output_folder = "vti_frames"
extension = ".vti"
Nx, Ny, Nt = 64, 64, 480
dx, dy, du = 1 / Nx, 1 / Ny, 1
dt = 3.0 / (Nt - 1)  # timestep

# Create folder to store vti
os.makedirs(output_folder, exist_ok=True)

# Load .mat file
data = loadmat(mat_file)
U_flat = data["U_flat"][:, 0]  # shape: (Nx*Ny*Nt, )
U = U_flat.reshape((Nx, Ny, Nt), order='F')  # (x, y, t)

# Output .vti for each timestep
for i in range(Nt):
    u_slice = U[:, :, i]

    # Create vtkImageData
    image = vtk.vtkImageData()
    image.SetDimensions(Nx, Ny, 1)
    image.SetSpacing(dx, dy, du)
    vtk_array = numpy_to_vtk(num_array=u_slice.ravel(order='F'), deep=True)
    vtk_array.SetName("u")
    image.GetPointData().SetScalars(vtk_array)

    # write .vti
    filename = f"{i:03d}{extension}"
    full_path = os.path.join(output_folder, filename)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(full_path)
    writer.SetInputData(image)
    writer.Write()

print(f"Totally export {Nt} .vti file to folder: {output_folder}")

# generate .pvd
pvd_path = os.path.join("./", "heat_ground_truth.pvd")
pvd_content = """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
  <Collection>
"""

for i in range(Nt):
    time = i * dt
    filename = f"{i:03d}{extension}"
    pvd_content += f'    <DataSet timestep="{time:.6f}" group="" part="0" file="{output_folder}/{filename}"/>\n'

pvd_content += """  </Collection>
</VTKFile>
"""

with open(pvd_path, "w") as f:
    f.write(pvd_content)

print(f"Output .pvd file：{pvd_path}")