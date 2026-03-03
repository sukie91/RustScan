//! Loaders for mesh files (OBJ and ASCII PLY).
//!
//! Parses the exact formats produced by RustSLAM's mesh_io module.

use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::loader::checkpoint::LoadError;
use crate::renderer::scene::{MeshGpuVertex, Scene};

/// Load an OBJ mesh (RustSLAM format: `v x y z r g b`, `vn nx ny nz`, `f A//A B//B C//C`).
pub fn load_obj(path: &Path) -> Result<(Vec<MeshGpuVertex>, Vec<u32>), LoadError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new(); // each: [v_idx, n_idx] × 3, stored flat
    let mut face_normals: Vec<[usize; 3]> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("v") => {
                let vals: Result<Vec<f32>, _> = parts
                    .map(|s| s.parse::<f32>())
                    .collect();
                let vals = vals.map_err(|_| LoadError::ObjParse(format!("bad vertex line: {line}")))?;
                if vals.len() >= 3 {
                    positions.push([vals[0], vals[1], vals[2]]);
                    let r = vals.get(3).copied().unwrap_or(0.5);
                    let g = vals.get(4).copied().unwrap_or(0.5);
                    let b = vals.get(5).copied().unwrap_or(0.5);
                    colors.push([r, g, b]);
                }
            }
            Some("vn") => {
                let vals: Result<Vec<f32>, _> = parts
                    .map(|s| s.parse::<f32>())
                    .collect();
                let vals = vals.map_err(|_| LoadError::ObjParse(format!("bad normal line: {line}")))?;
                if vals.len() >= 3 {
                    normals.push([vals[0], vals[1], vals[2]]);
                }
            }
            Some("f") => {
                // Format: A//A B//B C//C (1-based)
                let corners: Vec<&str> = parts.collect();
                if corners.len() >= 3 {
                    let mut vidx = [0usize; 3];
                    let mut nidx = [0usize; 3];
                    for (i, corner) in corners[..3].iter().enumerate() {
                        let mut split = corner.split("//");
                        let v = split
                            .next()
                            .and_then(|s| s.parse::<usize>().ok())
                            .ok_or_else(|| LoadError::ObjParse(format!("bad face vertex: {corner}")))?;
                        let n = split
                            .next()
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(1);
                        vidx[i] = v.saturating_sub(1);
                        nidx[i] = n.saturating_sub(1);
                    }
                    faces.push(vidx);
                    face_normals.push(nidx);
                }
            }
            _ => {}
        }
    }

    // Build flat vertex + index buffers (one MeshGpuVertex per face corner)
    let mut vertices: Vec<MeshGpuVertex> = Vec::with_capacity(faces.len() * 3);
    let mut indices: Vec<u32> = Vec::with_capacity(faces.len() * 3);

    for (face, fnorm) in faces.iter().zip(face_normals.iter()) {
        for i in 0..3 {
            let vi = face[i];
            let ni = fnorm[i];
            let pos = *positions.get(vi).unwrap_or(&[0.0; 3]);
            let col = *colors.get(vi).unwrap_or(&[0.5; 3]);
            let nrm = *normals.get(ni).unwrap_or(&[0.0, 1.0, 0.0]);
            indices.push(vertices.len() as u32);
            vertices.push(MeshGpuVertex {
                position: pos,
                normal: nrm,
                color: col,
            });
        }
    }

    Ok((vertices, indices))
}

/// Load an ASCII PLY mesh (RustSLAM format).
pub fn load_ply(path: &Path) -> Result<(Vec<MeshGpuVertex>, Vec<u32>), LoadError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut vertex_count = 0usize;
    let mut face_count = 0usize;
    let mut in_header = true;

    // Parse header
    while in_header {
        let line = lines
            .next()
            .ok_or_else(|| LoadError::PlyParse("unexpected end of header".into()))??;
        let line = line.trim().to_string();
        if line == "end_header" {
            in_header = false;
        } else if line.starts_with("element vertex") {
            vertex_count = line
                .split_whitespace()
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        } else if line.starts_with("element face") {
            face_count = line
                .split_whitespace()
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        }
    }

    // Read vertices: x y z nx ny nz r g b
    let mut vertices: Vec<MeshGpuVertex> = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let line = lines
            .next()
            .ok_or_else(|| LoadError::PlyParse("unexpected end of vertex data".into()))??;
        let vals: Vec<f32> = line
            .split_whitespace()
            .map(|s| s.parse::<f32>().unwrap_or(0.0))
            .collect();
        if vals.len() < 9 {
            return Err(LoadError::PlyParse(format!(
                "vertex line too short: {}",
                line
            )));
        }
        vertices.push(MeshGpuVertex {
            position: [vals[0], vals[1], vals[2]],
            normal: [vals[3], vals[4], vals[5]],
            color: [vals[6] / 255.0, vals[7] / 255.0, vals[8] / 255.0],
        });
    }

    // Read faces: 3 i j k
    let mut indices: Vec<u32> = Vec::with_capacity(face_count * 3);
    for _ in 0..face_count {
        let line = lines
            .next()
            .ok_or_else(|| LoadError::PlyParse("unexpected end of face data".into()))??;
        let vals: Vec<u32> = line
            .split_whitespace()
            .skip(1) // skip vertex count (always 3)
            .map(|s| s.parse::<u32>().unwrap_or(0))
            .collect();
        if vals.len() < 3 {
            return Err(LoadError::PlyParse(format!("face line too short: {}", line)));
        }
        indices.extend_from_slice(&vals[..3]);
    }

    Ok((vertices, indices))
}

/// Load a mesh file (OBJ or PLY) and merge into the scene.
pub fn load_mesh(path: &Path, scene: &mut Scene) -> Result<(), LoadError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let (verts, idxs) = match ext.as_str() {
        "obj" => load_obj(path)?,
        "ply" => load_ply(path)?,
        other => {
            return Err(LoadError::UnsupportedFormat(format!(
                "unsupported mesh format: .{other}"
            )));
        }
    };

    for v in &verts {
        scene.bounds.extend(v.position);
    }
    scene.mesh_vertices = verts;
    scene.mesh_indices = idxs;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_load_obj_triangle() {
        let obj = "# rustslam mesh\n\
                   v 0.0 0.0 0.0 0.8 0.2 0.2\n\
                   v 1.0 0.0 0.0 0.2 0.8 0.2\n\
                   v 0.0 1.0 0.0 0.2 0.2 0.8\n\
                   vn 0.0 0.0 1.0\n\
                   vn 0.0 0.0 1.0\n\
                   vn 0.0 0.0 1.0\n\
                   f 1//1 2//2 3//3\n";

        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        tmpfile.write_all(obj.as_bytes()).unwrap();
        let path = tmpfile.path().to_path_buf();

        let (verts, idxs) = load_obj(&path).unwrap();
        assert_eq!(verts.len(), 3, "should have 3 vertices");
        assert_eq!(idxs, vec![0, 1, 2], "indices should be 0,1,2");
        let c = verts[0].color;
        assert!((c[0] - 0.8).abs() < 1e-3, "red component mismatch: {}", c[0]);
    }

    #[test]
    fn test_load_ply_ascii() {
        let ply = "ply\n\
                   format ascii 1.0\n\
                   comment rustslam_mesh\n\
                   element vertex 3\n\
                   property float x\n\
                   property float y\n\
                   property float z\n\
                   property float nx\n\
                   property float ny\n\
                   property float nz\n\
                   property uchar red\n\
                   property uchar green\n\
                   property uchar blue\n\
                   element face 1\n\
                   property list uchar int vertex_indices\n\
                   end_header\n\
                   0.0 0.0 0.0 0.0 0.0 1.0 204 51 51\n\
                   1.0 0.0 0.0 0.0 0.0 1.0 51 204 51\n\
                   0.0 1.0 0.0 0.0 0.0 1.0 51 51 204\n\
                   3 0 1 2\n";

        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        tmpfile.write_all(ply.as_bytes()).unwrap();
        let path = tmpfile.path().to_path_buf();

        let (verts, idxs) = load_ply(&path).unwrap();
        assert_eq!(verts.len(), 3);
        assert_eq!(idxs, vec![0, 1, 2]);
        let c = verts[0].color;
        assert!((c[0] - 204.0 / 255.0).abs() < 1e-3, "red: {}", c[0]);
        assert!((c[1] - 51.0 / 255.0).abs() < 1e-3, "green: {}", c[1]);
    }
}
