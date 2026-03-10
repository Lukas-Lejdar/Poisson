
import pyvista as pv
import numpy as np
import natsort
import glob
import time

#for file in sorted(glob.glob("build/solutions/*.vtu")):
for file in sorted(glob.glob("build/reactor_solutions_0.8adaptive/*.vtu")):
#for file in sorted(glob.glob("build/winslow_solutions/*.vtu")):
    mesh = pv.read(file)

    potential = mesh["potential_(V)"]
    error_per_cell = mesh["error_per_cell"]
    potential_diff = mesh["potential_diff_(V)"]

    Ediff = mesh["E_diff"]
    E = mesh["E"]

    #mesh.points[:, 2] = E[:, 0]
    #mesh.points[:, 2] = potential
    
    cells = mesh.cells.reshape(-1, 5)
    cells = cells[:, 1:]   # remove the leading "4"

    for cell_id, verts in enumerate(cells):
        xy = mesh.points[verts, :-1]

        A = np.zeros([3, 3])
        A[:, :-1] = mesh.points[verts[1:], :-1] - mesh.points[verts[0], :-1]
        A[:, -1] = E[verts[1:], 0] - E[verts[0], 0]

        mesh.points[verts, 2] = np.linalg.det(A)


    plotter = pv.Plotter()
    plotter.camera_position = [
        (0.0, 0.0, 5.0),
        (0.0, 0.0, 0.0),
        (0, 1, 0)
    ]
    
    mesh.points[:, 2] = E[:, 0]

    plotter.add_mesh(mesh, scalars="potential_(V)", show_edges=True, cmap="viridis")
    plotter.add_text(f"{file}", position='upper_left', font_size=12, color='black')
    plotter.show_axes()
    plotter.show(auto_close=False)
    plotter.close()

