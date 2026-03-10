import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from plyfile import PlyData
import glob

ply = PlyData.read("evaluator_vertices.ply")
vertices = ply['vertex']
evaluator_vertices = np.column_stack((
    vertices.data['x'],
    vertices.data['y'],
    vertices.data['z']
))

files = sorted(glob.glob("build/reactor_solutions_0.8adaptive/*.vtu"))
#files = [files[0]]
efield_mag_eval = np.zeros([ len(evaluator_vertices), len(files) ])

water_level = 2.5

potential_diff_max = []
efield_diff_mag_max_air = []
efield_diff_mag_max = []
efield_mag_max_air = []

tol = 1e-5
min_y = np.min(evaluator_vertices[:, 1]) - tol
max_y = np.max(evaluator_vertices[:, 1]) + tol
min_x = np.min(evaluator_vertices[:, 0]) - tol
max_x = np.max(evaluator_vertices[:, 0]) + tol

data = []

for file_idx, file in enumerate(files):
    mesh = pv.read(file)
    points = mesh.points

    potential = mesh["potential_(V)"]
    error_per_cell = mesh["error_per_cell"]
    potential_diff = mesh["potential_diff_(V)"]
    efield_diff = mesh["E_diff"]
    efield = mesh["E"]

    potential_diff_max.append(np.max(potential_diff))

    efield_mag = np.linalg.norm(efield, axis=1)
    efield_diff_mag = np.linalg.norm(efield_diff, axis=1)

    efield_diff_mag_max.append(np.max(efield_diff_mag))

    air_mask = points[:, 1] >= water_level
    efield_diff_mag_max_air.append(np.max(efield_diff_mag[air_mask]))
    efield_mag_max_air.append(np.max(efield_mag[air_mask]))

    eval_indices = np.where( np.isclose(points[:, 0], 0.330545) & air_mask)[0]
    sort_keys = (points[eval_indices, 1], efield_mag[eval_indices])
    order = np.lexsort(sort_keys)
    eval_indices = eval_indices[order]

    data.append([file, points[eval_indices, 1], efield_mag[eval_indices], potential[eval_indices]])

    print(file_idx)

    #plotter = pv.Plotter()
    #plotter.camera_position = [
    #    (0.0, 0.0, 5.0),
    #    (0.0, 0.0, 0.0),
    #    (0, 1, 0)
    #]
    #points[:, 2] = potential

    #plotter.add_mesh(mesh, scalars="potential_(V)", show_edges=True, cmap="viridis")
    #plotter.add_text(f"{file}", position='upper_left', font_size=12, color='black')
    #plotter.show_axes()
    #plotter.show(auto_close=False)
    #plotter.close()

for i in range(len(data)):
    plt.plot(data[i][1], data[i][2], marker='o', markersize=1.1, label=data[i][0])

plt.title(f"E field magnitude")
plt.legend()
plt.show()

data_array = []

for i in range(len(data)):
    inds =  np.ones_like(data[i][1]) * i
    stack = np.column_stack([inds, data[i][1], data[i][2], data[i][3]])
    data_array.append(stack)

data_array = np.vstack(data_array)
mask = np.isclose(data_array[:, 1], 2.5)
plt.scatter(data_array[mask, 0], data_array[mask, 2])
plt.show()

data_array = np.vstack(data_array)
mask = np.isclose(data_array[:, 1], 2.63889)
plt.scatter(data_array[mask, 0], data_array[mask, 2])
plt.show()

for i in range(len(data)):
    plt.plot(data[i][1], data[i][2], label=data[i][0])

plt.title(f"potential")
plt.legend()
plt.show()

coarse_verts = data[i][1]

potential_diff_max.pop(0)
efield_diff_mag_max.pop(0)
efield_diff_mag_max_air.pop(0)

plt.plot(np.arange(len(potential_diff_max))+1, potential_diff_max)
plt.title("max potential diff")
plt.show()

plt.plot(np.arange(len(efield_diff_mag_max))+1, efield_diff_mag_max)
plt.title("max E diff mag")
plt.show()

plt.plot(np.arange(len(efield_diff_mag_max_air))+1, efield_diff_mag_max_air)
plt.title("max E diff magnitude air")
plt.show()

plt.plot(np.arange(len(efield_mag_max_air)), efield_mag_max_air)
plt.title("max E mag air")
plt.show()




