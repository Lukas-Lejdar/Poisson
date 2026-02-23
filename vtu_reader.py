import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import glob

efield_diff_mag_max = []

for file in sorted(glob.glob("build/reactor_solutions/*.vtu")):
    mesh = pv.read(file)

    potential = mesh["potential_(V)"]
    error_per_cell = mesh["error_per_cell"]
    potential_diff = mesh["potential_diff_(V)"]
    efield_diff = mesh["E_diff"]
    efield = mesh["E"]

    efield_diff_mag = np.linalg.norm(efield_diff, axis=1)
    efield_diff_mag_max.append(np.max(efield_diff_mag))


efield_diff_mag_max.pop(0)
plt.plot(np.arange(len(efield_diff_mag_max))+1, efield_diff_mag_max)
plt.show()




