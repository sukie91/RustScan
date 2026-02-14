//! OBJ File Format Support
//!
//! Wavefront OBJ format is one of the most widely supported mesh formats.
//!
//! Format specification:
//! - v x y z          (vertex position)
//! - vn nx ny nz      (vertex normal)
//! - vt u v           (texture coordinate)
//! - f v1 v2 v3       (face with vertex indices)
//! - f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 (face with all attributes)

use crate::RustMesh;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Write mesh to OBJ file
pub fn write_obj(mesh: &RustMesh, path: impl AsRef<Path>) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "# RustMesh OBJ Export")?;
    writeln!(writer, "# Vertices: {}", mesh.n_vertices())?;
    writeln!(writer, "# Faces: {}", mesh.n_faces())?;
    writeln!(writer)?;

    // Write vertices
    for v_idx in 0..mesh.n_vertices() {
        if let Some(point) = mesh.point_by_index(v_idx) {
            writeln!(writer, "v {} {} {}", point.x, point.y, point.z)?;
        }
    }

    writeln!(writer)?;

    // Write normals if available
    let has_normals = mesh.has_vertex_normals();
    if has_normals {
        for v_idx in 0..mesh.n_vertices() {
            if let Some(normal) = mesh.vertex_normal_by_index(v_idx) {
                writeln!(writer, "vn {} {} {}", normal.x, normal.y, normal.z)?;
            }
        }
        writeln!(writer)?;
    }

    // Write texture coordinates if available
    let has_texcoords = mesh.has_vertex_texcoords();
    if has_texcoords {
        for v_idx in 0..mesh.n_vertices() {
            if let Some(tc) = mesh.vertex_texcoord_by_index(v_idx) {
                writeln!(writer, "vt {} {}", tc.x, tc.y)?;
            }
        }
        writeln!(writer)?;
    }

    // Write faces
    for f_idx in 0..mesh.n_faces() {
        let face_handle = crate::FaceHandle::new(f_idx as u32);
        let vertices = mesh.face_vertices_vec(face_handle);

        if vertices.is_empty() {
            continue;
        }

        write!(writer, "f")?;

        for vh in vertices {
            let v = vh.idx_usize() + 1; // OBJ indices start at 1

            if has_normals && has_texcoords {
                // Format: f v/vt/vn
                write!(writer, " {}/{}/{}", v, v, v)?;
            } else if has_normals {
                // Format: f v//vn
                write!(writer, " {}//{}", v, v)?;
            } else if has_texcoords {
                // Format: f v/vt
                write!(writer, " {}/{}", v, v)?;
            } else {
                // Format: f v
                write!(writer, " {}", v)?;
            }
        }

        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

/// Read mesh from OBJ file
pub fn read_obj(path: impl AsRef<Path>) -> io::Result<RustMesh> {
    use std::io::{BufRead, BufReader};

    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut mesh = RustMesh::new();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut texcoords = Vec::new();

    // First pass: collect vertices
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "v" if parts.len() >= 4 => {
                let x = parts[1].parse::<f32>().unwrap_or(0.0);
                let y = parts[2].parse::<f32>().unwrap_or(0.0);
                let z = parts[3].parse::<f32>().unwrap_or(0.0);
                positions.push(glam::Vec3::new(x, y, z));
            }
            "vn" if parts.len() >= 4 => {
                let nx = parts[1].parse::<f32>().unwrap_or(0.0);
                let ny = parts[2].parse::<f32>().unwrap_or(0.0);
                let nz = parts[3].parse::<f32>().unwrap_or(0.0);
                normals.push(glam::Vec3::new(nx, ny, nz));
            }
            "vt" if parts.len() >= 3 => {
                let u = parts[1].parse::<f32>().unwrap_or(0.0);
                let v = parts[2].parse::<f32>().unwrap_or(0.0);
                texcoords.push(glam::Vec2::new(u, v));
            }
            _ => {}
        }
    }

    // Add vertices to mesh
    for pos in &positions {
        mesh.add_vertex(*pos);
    }

    // Request attributes if available
    if !normals.is_empty() {
        mesh.request_vertex_normals();
        for (i, normal) in normals.iter().enumerate() {
            if i < mesh.n_vertices() {
                mesh.set_vertex_normal_by_index(i, *normal);
            }
        }
    }

    if !texcoords.is_empty() {
        mesh.request_vertex_texcoords();
        for (i, tc) in texcoords.iter().enumerate() {
            if i < mesh.n_vertices() {
                mesh.set_vertex_texcoord_by_index(i, *tc);
            }
        }
    }

    // Second pass: read faces
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() || parts[0] != "f" {
            continue;
        }

        // Parse face indices
        let mut face_verts = Vec::new();

        for i in 1..parts.len() {
            let indices: Vec<&str> = parts[i].split('/').collect();
            if let Ok(v_idx) = indices[0].parse::<usize>() {
                if v_idx > 0 && v_idx <= mesh.n_vertices() {
                    face_verts.push(crate::VertexHandle::new((v_idx - 1) as u32));
                }
            }
        }

        if face_verts.len() >= 3 {
            mesh.add_face(&face_verts);
        }
    }

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obj_roundtrip() {
        use std::fs;

        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(glam::Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::Vec3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        let path = "/tmp/test_mesh.obj";
        write_obj(&mesh, path).unwrap();

        let loaded = read_obj(path).unwrap();
        assert_eq!(loaded.n_vertices(), 3);
        assert_eq!(loaded.n_faces(), 1);

        fs::remove_file(path).ok();
    }
}
