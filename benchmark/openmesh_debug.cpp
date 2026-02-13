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
              << ", E=" << mesh.n_edges()
              << ", F=" << mesh.n_faces() 
              << ", H=" << mesh.n_halfedges() << std::endl;
    
    // Check halfedge 0
    std::cout << "\n=== Checking halfedge 0 ===" << std::endl;
    auto heh0 = Mesh::HalfedgeHandle(0);
    
    std::cout << "to_vertex: " << mesh.to_vertex_handle(heh0).idx() << std::endl;
    std::cout << "from_vertex: " << mesh.from_vertex_handle(heh0).idx() << std::endl;
    std::cout << "is_valid: " << heh0.is_valid() << std::endl;
    
    // Check is_collapse_ok
    std::cout << "\nis_collapse_ok..." << std::flush;
    bool ok = mesh.is_collapse_ok(heh0);
    std::cout << " result: " << ok << std::endl;
    
    // Check face handles
    std::cout << "\nface_handle: " << mesh.face_handle(heh0).idx() << std::endl;
    auto heh0_opp = mesh.opposite_halfedge_handle(heh0);
    std::cout << "opposite: " << heh0_opp.idx() << std::endl;
    std::cout << "opposite face: " << mesh.face_handle(heh0_opp).idx() << std::endl;
    
    return 0;
}
