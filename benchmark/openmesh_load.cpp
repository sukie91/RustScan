#include <iostream>
#include <chrono>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

typedef OpenMesh::PolyMesh_ArrayKernelT<> Mesh;

int main() {
    Mesh mesh;
    
    std::cout << "=== Loading with OpenMesh (source build) ===" << std::endl;
    std::string path = "../test_data/large/FinalBaseMesh.obj";
    std::cout << "Path: " << path << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (!OpenMesh::IO::read_mesh(mesh, path)) {
        std::cerr << "Error loading!" << std::endl;
        return 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n=== Statistics ===" << std::endl;
    std::cout << "Vertices: " << mesh.n_vertices() << std::endl;
    std::cout << "Edges: " << mesh.n_edges() << std::endl;
    std::cout << "Faces: " << mesh.n_faces() << std::endl;
    std::cout << "Halfedges: " << mesh.n_halfedges() << std::endl;
    std::cout << "Load time: " << ms.count() << " ms" << std::endl;
    
    // Test vertex circulator
    std::cout << "\n=== Vertex Circulator Test ===" << std::endl;
    int count = 0;
    for (auto vh : mesh.vertices()) {
        if (count >= 10) break;
        int neighbors = 0;
        for (auto vv_it = mesh.vv_begin(vh); vv_it != mesh.vv_end(vh); ++vv_it) {
            neighbors++;
        }
        std::cout << "Vertex " << vh.idx() << ": " << neighbors << " neighbors" << std::endl;
        count++;
    }
    
    // Test boundary
    std::cout << "\n=== Boundary Test ===" << std::endl;
    int boundary = 0;
    int sample = 0;
    for (auto heh : mesh.halfedges()) {
        if (sample >= 100) break;
        if (mesh.is_boundary(heh)) boundary++;
        sample++;
    }
    std::cout << "Boundary halfedges (sample 100): " << boundary << std::endl;
    
    return 0;
}
