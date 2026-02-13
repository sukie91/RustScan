#include <iostream>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

typedef OpenMesh::TriMesh_ArrayKernelT<> Mesh;

int main() {
    std::string path = "/Users/tfjiang/Projects/RustMesh/test_data/middle/cube.obj";
    
    Mesh mesh;
    if (!OpenMesh::IO::read_mesh(mesh, path)) {
        std::cerr << "Error loading!" << std::endl;
        return 1;
    }
    
    std::cout << "TriMesh: V=" << mesh.n_vertices() 
              << ", F=" << mesh.n_faces() 
              << ", H=" << mesh.n_halfedges() << std::endl;
    
    // Try collapse on halfedge 0
    std::cout << "is_collapse_ok(0)..." << std::flush;
    bool ok = mesh.is_collapse_ok(Mesh::HalfedgeHandle(0));
    std::cout << " " << ok << std::endl;
    
    return 0;
}
