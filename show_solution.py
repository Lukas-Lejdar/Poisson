
import pyvista as pv
import numpy as np
import natsort
import glob
import time

#for file in sorted(glob.glob("build/solutions/*.vtu")):
for file in sorted(glob.glob("build/reactor_solutions/*.vtu")):
#for file in sorted(glob.glob("build/winslow_solutions/*.vtu")):
    mesh = pv.read(file)
    mesh.points[:, 2] = mesh["potential_(V)"]

    plotter = pv.Plotter()
    plotter.camera_position = [
        (0.0, 0.0, 5.0),
        (0.0, 0.0, 0.0),
        (0, 1, 0)
    ]

    plotter.add_mesh(mesh, scalars="potential_(V)", show_edges=True, cmap="viridis")
    plotter.add_text(f"{file}", position='upper_left', font_size=12, color='black')
    plotter.show_axes()
    plotter.show(auto_close=False)
    plotter.close()

