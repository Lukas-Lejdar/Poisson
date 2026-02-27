import numpy as np
from plyfile import PlyData

ply = PlyData.read("reactor.ply")
vertices = ply['vertex']
attributes = list(vertices.data.dtype.names)
attributes.remove('x')
attributes.remove('y')
attributes.remove('z')

xyz = np.column_stack((
    vertices.data['x'],
    vertices.data['y'],
    vertices.data['z']
))

faces = ply['face']['vertex_indices']

material_ids = np.zeros(len(faces), dtype=int)
manifold_ids = []

for i, face in enumerate(faces):
    face_vertices = vertices.data[face]
    for j, attr in enumerate(attributes):
        active = np.all(face_vertices[attr])
        if not active:
            continue

        if attr.endswith('_mat'):
            material_ids[i] = j+1

        if attr.endswith('_manifold'):
            manifold_ids.append([i, j+1])


edges = set()
for face in faces:
    n = len(face)
    for i in range(n):
        edge = tuple(sorted((face[i], face[(i + 1) % n])))
        edges.add(edge)


boundary_ids = []
b_manifold_ids = []


for edge in edges:
    edge_vertices = vertices.data[np.array(edge)]

    for j, attr in enumerate(attributes):
        active = np.all(np.array(edge_vertices[attr]))
        if not active:
            continue

        if attr.endswith('_boundary'):
            boundary_ids.append([int(edge[0]), int(edge[1]), j+1])

        if attr.endswith('_bmanifold'):
            b_manifold_ids.append([int(edge[0]), int(edge[1]), j+1])


ply = PlyData.read("circle_centers.ply")
centers = ply['vertex']
center_attribs = list(centers.data.dtype.names)
center_attribs.remove('x')
center_attribs.remove('y')
center_attribs.remove('z')

circle_centers = []

for center in centers:
    for i, cattr in enumerate(center_attribs):
        if not center[cattr]:
            continue

        for j, attr in enumerate(attributes):
            if attr == cattr:
                circle_centers.append([
                    j+1, [float(center['x']), float(center['y']) ]
                ])

with open("src/mesh.h", "w", encoding="utf-8") as f:
    f.write('\n#pragma once\n\n')

    f.write("#include <vector>\n")

    f.write("\n")
    for j, att in enumerate(attributes):
        f.write(f"const int {att.upper()}_ID = {j+1};\n")

    f.write("\nconst std::vector<std::vector<float>> vertices = {\n")
    for x, y, z in xyz:
        f.write(f"    {{{x:.6f}f, {y:.6f}f}},\n")
    f.write("};\n")

    f.write("\nconst std::vector<std::vector<int>> faces = {\n")
    for i, face in enumerate(faces):
        v0, v1, v2, v3 = face
        f.write(f"    {{{v0}, {v1}, {v2}, {v3}}},\n")
    f.write("};\n")

    f.write("\nconst std::vector<int> material_ids = {\n    ")
    for id in material_ids:
        f.write(f"{id}, ")
    f.write("\n};\n")

    f.write("\nconst std::vector<std::pair<int, int>> manifold_ids = {\n")
    for idx, id in manifold_ids:
        f.write(f"{{ {idx}, {id} }}, ")
    f.write("\n};\n")

    f.write("\nconst std::vector<std::pair<int, std::vector<float>>> circle_centers = {\n")
    for id, center in circle_centers:
        f.write(f"    {{ {id}, {{ {center[0]}, {center[1]} }} }},\n")
    f.write("};\n")

    f.write("\nconst std::vector<std::pair<std::vector<int>, int>> boundary_ids = {\n")
    for v0, v1, id in boundary_ids:
        f.write(f"    {{{{{v0}, {v1} }}, {id}}},\n")
    f.write("};\n")

    f.write("\nconst std::vector<std::pair<std::vector<int>, int>> boundary_manifold_ids = {\n")
    for v0, v1, id in b_manifold_ids:
        f.write(f"    {{{{{v0}, {v1}}}, {id} }},\n")
    f.write("};\n")


    for j, attr in enumerate(attributes):
        if not attr.endswith('_vertices'): continue

        verts = np.where(vertices.data[attr])[0]
        f.write(f"\nconst std::vector<int> {attr[:-9]}_vertex_ids = {{\n    ")
        for v in verts:
            f.write(f"{v}, ")
        f.write("\n};\n")

print()
