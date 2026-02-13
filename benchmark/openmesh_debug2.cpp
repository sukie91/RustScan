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
    
    std::cout << "Loaded: V=" << mesh.n_vertices() 
              << ", H=" << mesh.n_halfedges() << std::endl;
    
    // Find first VALID halfedge
    std::cout << "\n=== Finding valid halfedge ===" << std::endl;
    Mesh::HalfedgeHandle valid_heh;
    bool found = false;
    
    for (int i = 0; i < 100 && !found; ++i) {
        auto heh = Mesh::HalfedgeHandle(i);
        if (!heh.is_valid()) continue;
        
        // Check if to_vertex and from_vertex are valid
        auto to_vh = mesh.to_vertex_handle(heh);
        auto from_vh = mesh.from_vertex_handle(heh);
        
        if (to_vh.is_valid() && from_vh.is_valid()) {
            // Check if vertices have valid halfedge handles
            auto v0_heh = mesh.halfedge_handle(to_vh);
            auto v1_heh = mesh.halfedge_handle(from_vh);
            
            if (v0_heh.is_valid() && v1_heh.is_valid()) {
                valid_heh = heh;
                found = true;
                std::cout << "Found valid heh " << i << ": to=" << to_vh.idx() 
                          << ", from=" << from_vh.idx() << std::endl;
            }
        }
    }
    
    if (!found) {
        std::cout << "No valid halfedge found!" << std::endl;
        return 0;
    }
    
    // Try is_collapse_ok on this
    std::cout << "\nis_collapse_ok(" << valid_heh.idx() << ")..." << std::flush;
    bool ok = mesh.is_collapse_ok(valid_heh);
    std::cout << " result: " << ok << std::endl;
    
    return 0;
}
