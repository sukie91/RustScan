#include <iostream>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

typedef OpenMesh::PolyMesh_ArrayKernelT<> Mesh;

int main() {
    std::string path = "/Users/tfjiang/Projects/RustMesh/test_data/middle/cube.obj";
    
    Mesh mesh;
    
    if (!OpenMesh::IO::read_mesh(mesh, path)) {
        std::cerr << "Error loading!" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded: V=" << mesh.n_vertices() << ", F=" << mesh.n_faces() << std::endl;
    std::cout << "Halfedges: " << mesh.n_halfedges() << std::endl;
    
    // Check first 10 halfedges
    std::cout << "\nChecking first 10 halfedges:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        auto heh = Mesh::HalfedgeHandle(i);
        bool is_ok = mesh.is_collapse_ok(heh);
        auto to_vh = mesh.to_vertex_handle(heh);
        auto from_vh = mesh.from_vertex_handle(heh);
        std::cout << "  heh " << i << ": to=" << to_vh.idx() << ", from=" << from_vh.idx() 
                  << ", is_collapse_ok=" << is_ok << std::endl;
    }
    
    return 0;
}
