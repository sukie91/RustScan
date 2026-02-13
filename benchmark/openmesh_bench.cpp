#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <iostream>
#include <chrono>

typedef OpenMesh::PolyMesh_ArrayKernelT<> Mesh;

int main() {
    Mesh mesh;
    
    std::cout << "=== Loading cube OFF with OpenMesh ===" << std::endl;
    
    if (!OpenMesh::IO::read_mesh(mesh, "/tmp/cube.off")) {
        std::cerr << "Error loading mesh" << std::endl;
        std::cerr << "Available readers:" << std::endl;
        // List available IO modules
        return 1;
    }
    
    std::cout << "Vertices: " << mesh.n_vertices() << std::endl;
    std::cout << "Faces: " << mesh.n_faces() << std::endl;
    
    return 0;
}
