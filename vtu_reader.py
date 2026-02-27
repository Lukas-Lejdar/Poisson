import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import glob

efield_diff_mag_max_air = []
efield_diff_mag_max = []
efield_mag_max_air = []

for file in sorted(glob.glob("build/reactor_solutions/*.vtu")):
    mesh = pv.read(file)

    points = mesh.points

    potential = mesh["potential_(V)"]
    error_per_cell = mesh["error_per_cell"]
    potential_diff = mesh["potential_diff_(V)"]
    efield_diff = mesh["E_diff"]
    efield = mesh["E"]

    efield_mag = np.linalg.norm(efield, axis=1)
    efield_diff_mag = np.linalg.norm(efield_diff, axis=1)

    efield_diff_mag_max.append(np.max(efield_diff_mag))

    air_mask = points[:, 1] >= 3.;
    efield_diff_mag_max_air.append(np.max(efield_diff_mag[air_mask]))
    efield_mag_max_air.append(np.max(efield_mag[air_mask]))


efield_diff_mag_max.pop(0)
efield_diff_mag_max_air.pop(0)
#plt.plot(np.arange(len(efield_diff_mag_max))+1, efield_diff_mag_max)

#plt.plot(np.arange(len(efield_diff_mag_max_air))+1, efield_diff_mag_max_air)
plt.plot(np.arange(len(efield_mag_max_air)), efield_mag_max_air)
plt.show()




