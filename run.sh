gmsh capacitor.geo -2 -format msh2 -o capacitor.msh
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make run


