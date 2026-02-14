//! PLY File Format Support
//!
//! Stanford PLY (Polygon File Format) supports both ASCII and binary formats.
//!
//! Format specification:
//! - Header with element definitions
//! - Vertex list (x, y, z, nx, ny, nz, red, green, blue)
//! - Face list (vertex count + indices)

use crate::RustMesh;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// PLY file format variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlyFormat {
    /// ASCII text format (human-readable)
    Ascii,
    /// Binary little-endian format
    BinaryLittleEndian,
    /// Binary big-endian format
    BinaryBigEndian,
}

/// Write mesh to PLY file
pub fn write_ply(
    mesh: &RustMesh,
    path: impl AsRef<Path>,
    format: PlyFormat,
) -> io::Result<()> {
    match format {
        PlyFormat::Ascii => write_ply_ascii(mesh, path),
        PlyFormat::BinaryLittleEndian => write_ply_binary(mesh, path, false),
        PlyFormat::BinaryBigEndian => write_ply_binary(mesh, path, true),
    }
}

/// Write PLY in ASCII format
fn write_ply_ascii(mesh: &RustMesh, path: impl AsRef<Path>) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let has_normals = mesh.has_vertex_normals();
    let has_colors = mesh.has_vertex_colors();

    // Write header
    writeln!(writer, "ply")?;
    writeln!(writer, "format ascii 1.0")?;
    writeln!(writer, "comment RustMesh PLY Export")?;
    writeln!(writer, "element vertex {}", mesh.n_vertices())?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;

    if has_normals {
        writeln!(writer, "property float nx")?;
        writeln!(writer, "property float ny")?;
        writeln!(writer, "property float nz")?;
    }

    if has_colors {
        writeln!(writer, "property uchar red")?;
        writeln!(writer, "property uchar green")?;
        writeln!(writer, "property uchar blue")?;
        writeln!(writer, "property uchar alpha")?;
    }

    writeln!(writer, "element face {}", mesh.n_faces())?;
    writeln!(writer, "property list uchar int vertex_indices")?;
    writeln!(writer, "end_header")?;

    // Write vertices
    for v_idx in 0..mesh.n_vertices() {
        if let Some(point) = mesh.point_by_index(v_idx) {
            write!(writer, "{} {} {}", point.x, point.y, point.z)?;

            if has_normals {
                if let Some(normal) = mesh.vertex_normal_by_index(v_idx) {
                    write!(writer, " {} {} {}", normal.x, normal.y, normal.z)?;
                } else {
                    write!(writer, " 0 0 1")?;
                }
            }

            if has_colors {
                if let Some(color) = mesh.vertex_color_by_index(v_idx) {
                    let r = (color.x.clamp(0.0, 1.0) * 255.0) as u8;
                    let g = (color.y.clamp(0.0, 1.0) * 255.0) as u8;
                    let b = (color.z.clamp(0.0, 1.0) * 255.0) as u8;
                    let a = (color.w.clamp(0.0, 1.0) * 255.0) as u8;
                    write!(writer, " {} {} {} {}", r, g, b, a)?;
                } else {
                    write!(writer, " 255 255 255 255")?;
                }
            }

            writeln!(writer)?;
        }
    }

    // Write faces
    for f_idx in 0..mesh.n_faces() {
        let face_handle = crate::FaceHandle::new(f_idx as u32);
        let vertices = mesh.face_vertices_vec(face_handle);

        if vertices.is_empty() {
            continue;
        }

        write!(writer, "{}", vertices.len())?;
        for vh in vertices {
            write!(writer, " {}", vh.idx_usize())?;
        }
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

/// Write PLY in binary format
fn write_ply_binary(
    mesh: &RustMesh,
    path: impl AsRef<Path>,
    big_endian: bool,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let has_normals = mesh.has_vertex_normals();
    let has_colors = mesh.has_vertex_colors();

    // Write header (always ASCII)
    let format_str = if big_endian {
        "binary_big_endian"
    } else {
        "binary_little_endian"
    };

    writeln!(writer, "ply")?;
    writeln!(writer, "format {} 1.0", format_str)?;
    writeln!(writer, "comment RustMesh PLY Export")?;
    writeln!(writer, "element vertex {}", mesh.n_vertices())?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;

    if has_normals {
        writeln!(writer, "property float nx")?;
        writeln!(writer, "property float ny")?;
        writeln!(writer, "property float nz")?;
    }

    if has_colors {
        writeln!(writer, "property uchar red")?;
        writeln!(writer, "property uchar green")?;
        writeln!(writer, "property uchar blue")?;
        writeln!(writer, "property uchar alpha")?;
    }

    writeln!(writer, "element face {}", mesh.n_faces())?;
    writeln!(writer, "property list uchar int vertex_indices")?;
    writeln!(writer, "end_header")?;

    // Write vertices (binary)
    for v_idx in 0..mesh.n_vertices() {
        if let Some(point) = mesh.point_by_index(v_idx) {
            if big_endian {
                writer.write_all(&point.x.to_be_bytes())?;
                writer.write_all(&point.y.to_be_bytes())?;
                writer.write_all(&point.z.to_be_bytes())?;
            } else {
                writer.write_all(&point.x.to_le_bytes())?;
                writer.write_all(&point.y.to_le_bytes())?;
                writer.write_all(&point.z.to_le_bytes())?;
            }

            if has_normals {
                if let Some(normal) = mesh.vertex_normal_by_index(v_idx) {
                    if big_endian {
                        writer.write_all(&normal.x.to_be_bytes())?;
                        writer.write_all(&normal.y.to_be_bytes())?;
                        writer.write_all(&normal.z.to_be_bytes())?;
                    } else {
                        writer.write_all(&normal.x.to_le_bytes())?;
                        writer.write_all(&normal.y.to_le_bytes())?;
                        writer.write_all(&normal.z.to_le_bytes())?;
                    }
                } else {
                    let zeros = [0.0f32; 3];
                    for z in zeros {
                        let bytes = if big_endian {
                            z.to_be_bytes()
                        } else {
                            z.to_le_bytes()
                        };
                        writer.write_all(&bytes)?;
                    }
                }
            }

            if has_colors {
                if let Some(color) = mesh.vertex_color_by_index(v_idx) {
                    let r = (color.x.clamp(0.0, 1.0) * 255.0) as u8;
                    let g = (color.y.clamp(0.0, 1.0) * 255.0) as u8;
                    let b = (color.z.clamp(0.0, 1.0) * 255.0) as u8;
                    let a = (color.w.clamp(0.0, 1.0) * 255.0) as u8;
                    writer.write_all(&[r, g, b, a])?;
                } else {
                    writer.write_all(&[255u8, 255, 255, 255])?;
                }
            }
        }
    }

    // Write faces (binary)
    for f_idx in 0..mesh.n_faces() {
        let face_handle = crate::FaceHandle::new(f_idx as u32);
        let vertices = mesh.face_vertices_vec(face_handle);

        if vertices.is_empty() {
            continue;
        }

        // Write vertex count
        writer.write_all(&[vertices.len() as u8])?;

        // Write indices
        for vh in vertices {
            let idx = vh.idx_usize() as i32;
            if big_endian {
                writer.write_all(&idx.to_be_bytes())?;
            } else {
                writer.write_all(&idx.to_le_bytes())?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

/// Read mesh from PLY file (basic implementation)
pub fn read_ply(_path: impl AsRef<Path>) -> io::Result<RustMesh> {
    // TODO: Implement PLY reader
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "PLY reading not yet implemented",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ply_ascii_write() {
        use std::fs;

        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(glam::Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::Vec3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        let path = "/tmp/test_mesh.ply";
        write_ply(&mesh, path, PlyFormat::Ascii).unwrap();

        // Verify file exists
        assert!(std::path::Path::new(path).exists());

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_ply_binary_write() {
        use std::fs;

        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(glam::Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::Vec3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        let path = "/tmp/test_mesh_binary.ply";
        write_ply(&mesh, path, PlyFormat::BinaryLittleEndian).unwrap();

        assert!(std::path::Path::new(path).exists());

        fs::remove_file(path).ok();
    }
}
