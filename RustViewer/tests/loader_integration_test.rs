//! Integration tests for RustViewer loaders with actual test data.

use rust_viewer::loader::{checkpoint, mesh};
use rust_viewer::renderer::scene::Scene;
use std::path::Path;

#[test]
#[ignore = "requires checkpoint file at ../RustSLAM/output/checkpoints/pipeline.json"]
fn test_load_checkpoint_from_output() {
    let checkpoint_path = Path::new("../RustSLAM/output/checkpoints/pipeline.json");
    if !checkpoint_path.exists() {
        eprintln!("Skipping test: checkpoint file not found at {:?}", checkpoint_path);
        return;
    }

    let mut scene = Scene::default();
    let result = checkpoint::load_checkpoint(checkpoint_path, &mut scene);

    assert!(result.is_ok(), "Failed to load checkpoint: {:?}", result.err());

    // Check that we loaded some data
    println!("Loaded {} keyframes", scene.trajectory.len());
    println!("Loaded {} map points", scene.map_points.len());

    assert!(scene.trajectory.len() > 0, "Should have loaded keyframes");
}

#[test]
#[ignore = "requires test file at ../test_data/middle/cube.obj"]
fn test_load_obj_cube() {
    let obj_path = Path::new("../test_data/middle/cube.obj");
    if !obj_path.exists() {
        eprintln!("Skipping test: cube.obj not found at {:?}", obj_path);
        return;
    }

    let mut scene = Scene::default();
    let result = mesh::load_mesh(obj_path, &mut scene);

    assert!(result.is_ok(), "Failed to load cube.obj: {:?}", result.err());

    println!("Loaded {} vertices", scene.mesh_vertices.len());
    println!("Loaded {} indices", scene.mesh_indices.len());
    println!("Loaded {} edge indices", scene.mesh_edge_indices.len());

    assert!(scene.mesh_vertices.len() > 0, "Should have loaded vertices");
    assert!(scene.mesh_indices.len() > 0, "Should have loaded faces");
}

#[test]
#[ignore = "requires test file at ../test_data/large/FinalBaseMesh.obj"]
fn test_load_obj_finalbasemesh() {
    let obj_path = Path::new("../test_data/large/FinalBaseMesh.obj");
    if !obj_path.exists() {
        eprintln!("Skipping test: FinalBaseMesh.obj not found at {:?}", obj_path);
        return;
    }

    let mut scene = Scene::default();
    let result = mesh::load_mesh(obj_path, &mut scene);

    assert!(result.is_ok(), "Failed to load FinalBaseMesh.obj: {:?}", result.err());

    println!("Loaded {} vertices", scene.mesh_vertices.len());
    println!("Loaded {} indices", scene.mesh_indices.len());
    println!("Loaded {} edge indices", scene.mesh_edge_indices.len());

    assert!(scene.mesh_vertices.len() > 0, "Should have loaded vertices");
    assert!(scene.mesh_indices.len() > 0, "Should have loaded faces");

    // FinalBaseMesh.obj has 24461 vertices and 48918 faces
    assert!(scene.mesh_vertices.len() >= 24000, "Should have ~24k vertices");
}
