
import pyvista as pv
import numpy as np
import natsort
import glob
import time

for file in glob.glob("build/*.vtu"):
    mesh = pv.read(file)
    mesh.points[:, 2] = mesh["potential"]

    plotter = pv.Plotter()
    plotter.camera_position = [
        (0.0, 0.0, 5.0),
        (0.0, 0.0, 0.0),
        (0, 1, 0)
    ]

    plotter.add_mesh(mesh, scalars="potential", cmap="viridis", clim=[0, 1])
    plotter.show(auto_close=False)
    plotter.close()

