#include <iostream>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

typedef OpenMesh::PolyMesh_ArrayKernelT<> Mesh;

int main() {
    std::string path = "/Users/tfjiang/Projects/RustMesh/test_data/middle/cube.obj";
    
    Mesh mesh;
    if (!OpenMesh::IO::read_mesh(mesh, path)) return 1;
    
    auto heh = Mesh::HalfedgeHandle(0);
    
    std::cout << "Calling is_collapse_ok..." << std::endl;
    bool ok = mesh.is_collapse_ok(heh);
    std::cout << "Result: " << ok << std::endl;
    
    return 0;
}
