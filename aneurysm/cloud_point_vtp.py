import numpy as np
from physicsnemo.sym.geometry.tessellation import Tessellation
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk

if __name__ == "__main__":
    # STL path
    stl_path = "real_stl_files/aneurysm_closed.stl"

    # Read STL geometry
    geo = Tessellation.from_stl(stl_path, airtight=True)

    # number of points to sample
    nr_points = 100000

    # sample geometry for plotting in Paraview
    s = geo.sample_boundary(nr_points=nr_points)
    var_to_polyvtk(s, "aneurysm_closed_boundary")
    print("Surface Area: {:.3f}".format(np.sum(s["area"])))
    
    # Only generating cloud point, not need  SDF info
    s = geo.sample_interior(nr_points=nr_points, compute_sdf_derivatives=False)
    var_to_polyvtk(s, "aneurysm_closed_interior")
    print("Volume: {:.3f}".format(np.sum(s["area"])))