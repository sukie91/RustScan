#include <iostream>
#include <chrono>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

typedef OpenMesh::PolyMesh_ArrayKernelT<> Mesh;

int main() {
    std::string path = "/Users/tfjiang/Projects/RustMesh/test_data/middle/cube.obj";
    
    std::cout << "=== OpenMesh Loading ===" << std::endl;
    std::cout << "Path: " << path << std::endl;
    
    Mesh mesh;
    
    auto start = std::chrono::high_resolution_clock::now();
    if (!OpenMesh::IO::read_mesh(mesh, path)) {
        std::cerr << "Error loading!" << std::endl;
        return 1;
    }
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    
    int n_verts = mesh.n_vertices();
    int n_faces = mesh.n_faces();
    int n_edges = mesh.n_edges();
    
    std::cout << "Load time: " << load_time.count() << " ms" << std::endl;
    std::cout << "Original: V=" << n_verts << ", E=" << n_edges << ", F=" << n_faces << std::endl;
    
    // Test sequential decimation
    std::cout << "\n=== Sequential Decimation ===" << std::endl;
    int target = n_verts / 2;
    int collapse_count = n_verts - target;
    
    start = std::chrono::high_resolution_clock::now();
    
    int collapsed = 0;
    for (int i = 0; i < mesh.n_halfedges() && collapsed < collapse_count; ++i) {
        auto heh = Mesh::HalfedgeHandle(i);
        if (mesh.is_collapse_ok(heh)) {
            mesh.collapse(heh);
            collapsed++;
        }
    }
    
    auto decimate_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    
    std::cout << "Collapsed: " << collapsed << " edges" << std::endl;
    std::cout << "Time: " << decimate_time.count() << " ms" << std::endl;
    std::cout << "After: V=" << mesh.n_vertices() << ", F=" << mesh.n_faces() << std::endl;
    
    return 0;
}
