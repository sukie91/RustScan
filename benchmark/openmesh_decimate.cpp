#include <iostream>
#include <chrono>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

typedef OpenMesh::PolyMesh_ArrayKernelT<> Mesh;

int main() {
    Mesh mesh;
    
    // Use absolute path
    std::string path = "/Users/tfjiang/Projects/RustMesh/test_data/large/FinalBaseMesh.obj";
    
    std::cout << "Loading: " << path << std::endl;
    
    if (!OpenMesh::IO::read_mesh(mesh, path)) {
        std::cerr << "Error loading!" << std::endl;
        return 1;
    }
    
    int orig_verts = mesh.n_vertices();
    int orig_faces = mesh.n_faces();
    
    std::cout << "Original: V=" << orig_verts << ", F=" << orig_faces << std::endl;
    
    // Test with a few collapses
    std::cout << "\n=== Testing 10 collapses ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    int collapsed = 0;
    for (int i = 0; i < mesh.n_halfedges() && collapsed < 10; ++i) {
        auto heh = Mesh::HalfedgeHandle(i);
        if (mesh.is_collapse_ok(heh)) {
            std::cout << "Collapsing " << i << "..." << std::flush;
            mesh.collapse(heh);
            collapsed++;
            std::cout << " OK" << std::endl;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Collapsed: " << collapsed << " edges" << std::endl;
    std::cout << "Time: " << ms.count() << " ms" << std::endl;
    
    return 0;
}
