#include <iostream>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

typedef OpenMesh::PolyMesh_ArrayKernelT<> Mesh;

int main() {
    Mesh mesh;
    
    if (!OpenMesh::IO::read_mesh(mesh, "/tmp/simple_cube.off")) {
        std::cerr << "Error loading!" << std::endl;
        return 1;
    }
    
    std::cout << "Simple cube: V=" << mesh.n_vertices() 
              << ", F=" << mesh.n_faces() 
              << ", H=" << mesh.n_halfedges() << std::endl;
    
    // Try collapse on halfedge 0
    std::cout << "is_collapse_ok(0)..." << std::flush;
    bool ok = mesh.is_collapse_ok(Mesh::HalfedgeHandle(0));
    std::cout << " " << ok << std::endl;
    
    if (ok) {
        std::cout << "collapse(0)..." << std::flush;
        mesh.collapse(Mesh::HalfedgeHandle(0));
        std::cout << " done" << std::endl;
    }
    
    std::cout << "After: V=" << mesh.n_vertices() << ", F=" << mesh.n_faces() << std::endl;
    
    return 0;
}
