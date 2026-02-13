#include <iostream>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

typedef OpenMesh::PolyMesh_ArrayKernelT<> Mesh;

int main() {
    std::string path = "/Users/tfjiang/Projects/RustMesh/test_data/middle/cube.obj";
    
    Mesh mesh;
    if (!OpenMesh::IO::read_mesh(mesh, path)) return 1;
    
    auto heh = Mesh::HalfedgeHandle(0);
    auto v0 = mesh.to_vertex_handle(heh);
    auto v1 = mesh.from_vertex_handle(heh);
    
    std::cout << "v0=" << v0.idx() << ", v1=" << v1.idx() << std::endl;
    
    // Step 1: Check opposite
    std::cout << "1. opposite..." << std::flush;
    auto heh_opp = mesh.opposite_halfedge_handle(heh);
    std::cout << " " << heh_opp.idx() << std::endl;
    
    // Step 2: Check faces
    std::cout << "2. face_handle..." << std::flush;
    auto fh_left = mesh.face_handle(heh);
    auto fh_right = mesh.face_handle(heh_opp);
    std::cout << " left=" << fh_left.idx() << ", right=" << fh_right.idx() << std::endl;
    
    // Step 3: Check next halfedges
    std::cout << "3. next_halfedge..." << std::flush;
    auto next_left = mesh.next_halfedge_handle(heh);
    auto next_right = mesh.next_halfedge_handle(heh_opp);
    std::cout << " left=" << next_left.idx() << ", right=" << next_right.idx() << std::endl;
    
    // Step 4: Get vertex halfedge handles
    std::cout << "4. vertex halfedge handles..." << std::flush;
    auto v0_heh = mesh.halfedge_handle(v0);
    auto v1_heh = mesh.halfedge_handle(v1);
    std::cout << " v0=" << v0_heh.idx() << ", v1=" << v1_heh.idx() << std::endl;
    
    // Step 5: Check boundary
    std::cout << "5. is_boundary v0_heh..." << std::flush;
    bool b0 = mesh.is_boundary(v0_heh);
    std::cout << " " << b0 << std::endl;
    
    std::cout << "Done!" << std::endl;
    return 0;
}
