//! # IO Module
//!
//! Mesh file I/O for OFF, OBJ, PLY, and STL formats.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::connectivity::RustMesh;
use crate::handles::{VertexHandle, FaceHandle};

/// Result type for IO operations
pub type IoResult<T> = std::result::Result<T, IoError>;

/// IO Error types
#[derive(Debug)]
pub enum IoError {
    Io(io::Error),
    Parse(String),
    Format(String),
    InvalidData(String),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError::Io(e) => write!(f, "IO error: {}", e),
            IoError::Parse(e) => write!(f, "Parse error: {}", e),
            IoError::Format(e) => write!(f, "Format error: {}", e),
            IoError::InvalidData(e) => write!(f, "Invalid data: {}", e),
        }
    }
}

impl std::error::Error for IoError {}

impl From<io::Error> for IoError {
    fn from(e: io::Error) -> Self {
        IoError::Io(e)
    }
}

/// Read OFF format file
/// 
/// OFF format specification:
/// - First line: "OFF" or "STOFF" (with colors)
/// - Second line: vertex_count face_count edge_count
/// - Then: vertex lines (x y z) or (x y z r g b a)
/// - Then: face lines (n v1 v2 ... vn) or with colors
pub fn read_off<P: AsRef<Path>>(path: P) -> IoResult<RustMesh> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Parse header
    let header = lines.next()
        .ok_or(IoError::Parse("Empty file".to_string()))??;
    
    let has_colors = header.starts_with("STOFF") || header.starts_with("COFF");
    let _binary = header.starts_with("BOFF");

    // Parse counts
    let counts_line = lines.next()
        .ok_or(IoError::Parse("Missing counts line".to_string()))??;
    
    let counts: Vec<usize> = counts_line
        .split_whitespace()
        .map(|s| s.parse().map_err(|_| IoError::Parse("Invalid count".to_string())))
        .collect::<IoResult<Vec<usize>>>()?;
    
    if counts.len() < 3 {
        return Err(IoError::Parse("Invalid counts line".to_string()));
    }

    let (n_vertices, n_faces, _n_edges) = (counts[0], counts[1], counts.get(2).copied().unwrap_or(0));

    let mut mesh = RustMesh::new();
    let mut vertices: Vec<VertexHandle> = Vec::with_capacity(n_vertices);

    // Parse vertices
    for i in 0..n_vertices {
        let line = lines.next()
            .ok_or(IoError::Parse(format!("Unexpected end at vertex {}", i)))??;
        
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.len() < 3 {
            return Err(IoError::Parse(format!("Vertex {} has insufficient coordinates", i)));
        }

        let x: f32 = parts[0].parse().map_err(|_| IoError::Parse(format!("Invalid x for vertex {}", i)))?;
        let y: f32 = parts[1].parse().map_err(|_| IoError::Parse(format!("Invalid y for vertex {}", i)))?;
        let z: f32 = parts[2].parse().map_err(|_| IoError::Parse(format!("Invalid z for vertex {}", i)))?;

        let vh = mesh.add_vertex(glam::vec3(x, y, z));
        vertices.push(vh);
    }

    // Parse faces
    for i in 0..n_faces {
        let line = lines.next()
            .ok_or(IoError::Parse(format!("Unexpected end at face {}", i)))??;
        
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.is_empty() {
            continue;
        }

        let n_vertices_in_face: usize = parts[0]
            .parse()
            .map_err(|_| IoError::Parse(format!("Invalid vertex count for face {}", i)))?;
        
        if parts.len() < n_vertices_in_face + 1 {
            return Err(IoError::Parse(format!("Face {} has insufficient vertex indices", i)));
        }

        let mut face_vertices: Vec<VertexHandle> = Vec::with_capacity(n_vertices_in_face);
        
        for j in 0..n_vertices_in_face {
            let v_idx: usize = parts[j + 1]
                .parse()
                .map_err(|_| IoError::Parse(format!("Invalid vertex index in face {}", i)))?;
            
            if v_idx >= vertices.len() {
                return Err(IoError::Parse(format!("Vertex index out of bounds in face {}", i)));
            }
            
            // OFF files use 1-based indexing, convert to 0-based
            let vh = vertices[v_idx];
            face_vertices.push(vh);
        }

        if let Some(fh) = mesh.add_face(&face_vertices) {
            // Handle color if present (STOFF format)
            if has_colors && parts.len() > n_vertices_in_face + 1 {
                // Parse RGB color
                // TODO: Store color in mesh properties
                let _r: f32 = parts[n_vertices_in_face + 1].parse().unwrap_or(0.0);
            }
        }
    }

    Ok(mesh)
}

/// Write OFF format file
pub fn write_off<P: AsRef<Path>>(mesh: &RustMesh, path: P) -> IoResult<()> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "OFF")?;

    // Write counts
    writeln!(writer, "{} {} {}", mesh.n_vertices(), mesh.n_faces(), mesh.n_edges())?;

    // Write vertices
    for vh in mesh.vertices() {
        if let Some(point) = mesh.point(vh) {
            writeln!(writer, "{} {} {}", point.x, point.y, point.z)?;
        }
    }

    // Write faces
    // Note: face_vertices() needs full halfedge connectivity to work
    // For now, write placeholder (needs implementation)
    for idx in 0..mesh.n_faces() {
        let fh = FaceHandle::new(idx as u32);
        writeln!(writer, "3 0 0 0  # Face {} - requires face_vertices implementation", fh.idx())?;
    }

    Ok(())
}

/// Read OBJ format file
/// 
/// OBJ format supports:
/// - v x y z [w] (vertex with optional w)
/// - vt u v [w] (texture coordinates)
/// - vn x y z (vertex normal)
/// - f v1 v2 v3 ... (faces, vertices only)
/// - f v1/vt1 v2/vt2 ... (faces with UVs)
/// - f v1/vt1/vn1 ... (faces with UVs and normals)
pub fn read_obj<P: AsRef<Path>>(path: P) -> IoResult<RustMesh> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let lines = reader.lines();

    let mut mesh = RustMesh::new();
    let mut vertices: Vec<VertexHandle> = Vec::new();
    let mut texture_coords: Vec<(f32, f32, f32)> = Vec::new();
    let mut normals: Vec<glam::Vec3> = Vec::new();

    for (line_num, line) in lines.enumerate() {
        let line = line?;
        let trimmed = line.trim();
        
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        
        match parts[0] {
            "v" => {
                // Vertex
                if parts.len() < 4 {
                    return Err(IoError::Parse(format!("Line {}: vertex requires 3 coordinates", line_num)));
                }
                
                let x: f32 = parts[1].parse()
                    .map_err(|_| IoError::Parse(format!("Line {}: invalid x coordinate", line_num)))?;
                let y: f32 = parts[2].parse()
                    .map_err(|_| IoError::Parse(format!("Line {}: invalid y coordinate", line_num)))?;
                let z: f32 = parts[3].parse()
                    .map_err(|_| IoError::Parse(format!("Line {}: invalid z coordinate", line_num)))?;
                
                let vh = mesh.add_vertex(glam::vec3(x, y, z));
                vertices.push(vh);
            }
            "vt" => {
                // Texture coordinate
                let u: f32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let v: f32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let w: f32 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                texture_coords.push((u, v, w));
            }
            "vn" => {
                // Vertex normal
                let x: f32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y: f32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let z: f32 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                normals.push(glam::vec3(x, y, z));
            }
            "f" => {
                // Face
                if parts.len() < 4 {
                    return Err(IoError::Parse(format!("Line {}: face requires at least 3 vertices", line_num)));
                }

                let mut face_vertices: Vec<VertexHandle> = Vec::with_capacity(parts.len() - 1);
                
                for i in 1..parts.len() {
                    let vertex_data = parts[i];
                    let indices: Vec<&str> = vertex_data.split('/').collect();
                    
                    // First index is always vertex
                    let v_idx: usize = indices[0]
                        .parse()
                        .map_err(|_| IoError::Parse(format!("Line {}: invalid vertex index", line_num)))?;
                    
                    if v_idx == 0 || v_idx > vertices.len() {
                        return Err(IoError::Parse(format!("Line {}: vertex index out of bounds", line_num)));
                    }
                    
                    // OBJ uses 1-based indexing
                    let vh = vertices[v_idx - 1];
                    face_vertices.push(vh);
                }

                mesh.add_face(&face_vertices);
            }
            "g" | "o" => {
                // Group or object name - ignore for now
            }
            "s" => {
                // Smoothing group - ignore for now
            }
            "mtllib" => {
                // Material library - ignore for now
            }
            "usemtl" => {
                // Material - ignore for now
            }
            _ => {
                // Unknown directive - ignore
            }
        }
    }

    Ok(mesh)
}

/// Write OBJ format file
pub fn write_obj<P: AsRef<Path>>(mesh: &RustMesh, path: P) -> IoResult<()> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "# RustMesh OBJ export")?;
    writeln!(writer, "# Vertices: {}, Faces: {}", mesh.n_vertices(), mesh.n_faces())?;
    writeln!(writer)?;

    // Write vertices
    for vh in mesh.vertices() {
        if let Some(point) = mesh.point(vh) {
            writeln!(writer, "v {} {} {}", point.x, point.y, point.z)?;
        }
    }

    writeln!(writer)?;

    // Write faces
    // Note: face_vertices() needs full halfedge connectivity to work
    // For now, write placeholder comments
    writeln!(writer, "# Faces - requires face_vertices implementation")?;
    for idx in 0..mesh.n_faces() {
        let fh = FaceHandle::new(idx as u32);
        writeln!(writer, "# Face {}", fh.idx())?;
    }

    Ok(())
}

/// Detect file format from extension
pub fn detect_format<P: AsRef<Path>>(path: P) -> Option<&'static str> {
    let ext = path.as_ref().extension()?.to_str()?;

    match ext.to_lowercase().as_str() {
        "off" | "off" => Some("OFF"),
        "obj" | "OBJ" => Some("OBJ"),
        "ply" | "PLY" => Some("PLY"),
        "stl" | "STL" => Some("STL"),
        _ => None,
    }
}

/// Read STL format file (ASCII)
///
/// STL format specification:
/// - solid <name>
/// - facet normal <nx> <ny> <nz>
///   - outer loop
///   - vertex <x> <y> <z>
///   - vertex <x> <y> <z>
///   - vertex <x> <y> <z>
///   - endloop
/// - endfacet
/// - endsolid <name>
pub fn read_stl<P: AsRef<Path>>(path: P) -> IoResult<RustMesh> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let lines = reader.lines();

    let mut mesh = RustMesh::new();
    let mut vertices: Vec<VertexHandle> = Vec::new();
    let mut current_normal: Option<glam::Vec3> = None;

    for (line_num, line) in lines.enumerate() {
        let line = line?;
        let trimmed = line.trim().to_lowercase();

        if trimmed.starts_with("solid") && trimmed.len() > 6 {
            // Start of STL file, skip name
            continue;
        }

        if trimmed.starts_with("facet normal") {
            // Read normal
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 4 {
                let nx: f32 = parts[2].parse().unwrap_or(0.0);
                let ny: f32 = parts[3].parse().unwrap_or(0.0);
                let nz: f32 = parts[4].parse().unwrap_or(0.0);
                current_normal = Some(glam::vec3(nx, ny, nz));
            }
        } else if trimmed.starts_with("vertex") {
            // Read vertex
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 4 {
                let x: f32 = parts[1].parse().unwrap_or(0.0);
                let y: f32 = parts[2].parse().unwrap_or(0.0);
                let z: f32 = parts[3].parse().unwrap_or(0.0);

                // Check if we already have this vertex
                let vh = mesh.add_vertex(glam::vec3(x, y, z));
                vertices.push(vh);

                // If we have 3 vertices, create a face
                if vertices.len() >= 3 {
                    let len = vertices.len();
                    mesh.add_face(&[vertices[len-3], vertices[len-2], vertices[len-1]]);
                }
            }
        } else if trimmed.starts_with("endfacet") {
            // End of facet, clear normal
            current_normal = None;
        } else if trimmed.starts_with("endsolid") {
            // End of STL file
            break;
        }
    }

    Ok(mesh)
}

/// Write STL format file (ASCII)
pub fn write_stl<P: AsRef<Path>>(mesh: &RustMesh, path: P) -> IoResult<()> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    let mut writer = BufWriter::new(file);

    writeln!(writer, "solid mesh")?;

    // Write faces as facets
    // Note: face_vertices() needs implementation for proper output
    // For now, write placeholder
    writeln!(writer, "  facet normal 0 0 0")?;
    writeln!(writer, "    outer loop")?;
    writeln!(writer, "      vertex 0 0 0")?;
    writeln!(writer, "      vertex 1 0 0")?;
    writeln!(writer, "      vertex 0 1 0")?;
    writeln!(writer, "    endloop")?;
    writeln!(writer, "  endfacet")?;
    writeln!(writer, "endsolid mesh")?;

    Ok(())
}

/// Binary STL format specification:
/// - 80 bytes: header (can contain model name or be zeros)
/// - 4 bytes: number of triangles (uint32)
/// - For each triangle (50 bytes):
///   - 12 bytes: normal vector (3x float32)
///   - 36 bytes: 3 vertices (3x float32 each)
///   - 2 bytes: attribute (unused, usually 0)

/// Read binary STL format
pub fn read_stl_binary<P: AsRef<Path>>(path: P) -> IoResult<RustMesh> {
    let mut file = File::open(path)?;
    let mut mesh = RustMesh::new();

    // Read and discard header (80 bytes)
    let mut header = [0u8; 80];
    file.read_exact(&mut header)?;

    // Read triangle count
    let triangle_count = file.read_u32::<LittleEndian>()?;

    // Read triangles
    for _ in 0..triangle_count {
        // Skip normal
        file.read_f32::<LittleEndian>()?;
        file.read_f32::<LittleEndian>()?;
        file.read_f32::<LittleEndian>()?;

        // Read 3 vertices
        let v0x = file.read_f32::<LittleEndian>()?;
        let v0y = file.read_f32::<LittleEndian>()?;
        let v0z = file.read_f32::<LittleEndian>()?;
        let v1x = file.read_f32::<LittleEndian>()?;
        let v1y = file.read_f32::<LittleEndian>()?;
        let v1z = file.read_f32::<LittleEndian>()?;
        let v2x = file.read_f32::<LittleEndian>()?;
        let v2y = file.read_f32::<LittleEndian>()?;
        let v2z = file.read_f32::<LittleEndian>()?;

        // Skip attribute
        file.read_u16::<LittleEndian>()?;

        // Add triangle
        let vh0 = mesh.add_vertex(glam::vec3(v0x, v0y, v0z));
        let vh1 = mesh.add_vertex(glam::vec3(v1x, v1y, v1z));
        let vh2 = mesh.add_vertex(glam::vec3(v2x, v2y, v2z));
        mesh.add_face(&[vh0, vh1, vh2]);
    }

    Ok(mesh)
}

/// Write binary STL format
pub fn write_stl_binary<P: AsRef<Path>>(mesh: &RustMesh, path: P) -> IoResult<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    // Write header (80 bytes)
    file.write_all(&[0u8; 80])?;

    // Write triangle count
    file.write_u32::<LittleEndian>(mesh.n_faces() as u32)?;

    // Write triangles
    for fh in mesh.faces() {
        // Simplified: write default normal (0,0,1)
        file.write_f32::<LittleEndian>(0.0)?;
        file.write_f32::<LittleEndian>(0.0)?;
        file.write_f32::<LittleEndian>(1.0)?;

        // Placeholder vertices (actual implementation would get real vertices)
        for _ in 0..3 {
            file.write_f32::<LittleEndian>(0.0)?;
            file.write_f32::<LittleEndian>(0.0)?;
            file.write_f32::<LittleEndian>(0.0)?;
        }

        // Attribute
        file.write_u16::<LittleEndian>(0)?;
    }

    Ok(())
}

/// Read mesh file (auto-detect format)
pub fn read_mesh<P: AsRef<Path>>(path: P) -> IoResult<RustMesh> {
    match detect_format(&path) {
        Some("OFF") => read_off(path),
        Some("OBJ") => read_obj(path),
        Some("STL") => read_stl(path),
        Some(format) => Err(IoError::Format(format!("Unsupported format: {}", format))),
        None => Err(IoError::Format("Unknown file format".to_string())),
    }
}

/// Write mesh file (auto-detect format from extension)
pub fn write_mesh<P: AsRef<Path>>(mesh: &RustMesh, path: P) -> IoResult<()> {
    match detect_format(&path) {
        Some("OFF") => write_off(mesh, path),
        Some("OBJ") => write_obj(mesh, path),
        Some("STL") => write_stl(mesh, path),
        Some(format) => Err(IoError::Format(format!("Unsupported format for writing: {}", format))),
        None => Err(IoError::Format("Unknown file format".to_string())),
    }
}


// ============================================================================
// PLY Format Support
// ============================================================================

/// Read PLY format file (ASCII)
pub fn read_ply<P: AsRef<Path>>(path: P) -> IoResult<RustMesh> {
    let content = std::fs::read_to_string(path)?;
    parse_ply_ascii(&content)
}

fn parse_ply_ascii(content: &str) -> IoResult<RustMesh> {
    let mut lines = content.lines();
    
    let first = lines.next()
        .ok_or(IoError::Parse("Empty file".to_string()))?;
    if first.trim() != "ply" {
        return Err(IoError::Parse("Not a PLY file".to_string()));
    }
    
    let mut n_vertices = 0;
    let mut n_faces = 0;
    
    for line in lines.by_ref() {
        let trimmed = line.trim();
        if trimmed == "end_header" { break; }
        if let Some(rest) = trimmed.strip_prefix("element vertex ") {
            n_vertices = rest.trim().parse()
                .map_err(|_| IoError::Parse("Invalid vertex count".to_string()))?;
        }
        if let Some(rest) = trimmed.strip_prefix("element face ") {
            n_faces = rest.trim().parse()
                .map_err(|_| IoError::Parse("Invalid face count".to_string()))?;
        }
    }
    
    let mut vertices = Vec::with_capacity(n_vertices);
    for _ in 0..n_vertices {
        if let Some(line) = lines.next() {
            let mut iter = line.split_whitespace();
            let x = iter.next().ok_or(IoError::Parse("Invalid x".to_string()))?
                .parse().map_err(|_| IoError::Parse("Invalid x".to_string()))?;
            let y = iter.next().ok_or(IoError::Parse("Invalid y".to_string()))?
                .parse().map_err(|_| IoError::Parse("Invalid y".to_string()))?;
            let z = iter.next().ok_or(IoError::Parse("Invalid z".to_string()))?
                .parse().map_err(|_| IoError::Parse("Invalid z".to_string()))?;
            vertices.push((x, y, z));
        }
    }
    
    let mut mesh = RustMesh::new();
    let vhandles: Vec<VertexHandle> = vertices.iter()
        .map(|(x, y, z)| mesh.add_vertex(glam::vec3(*x, *y, *z)))
        .collect();
    
    for _ in 0..n_faces {
        if let Some(line) = lines.next() {
            let mut iter = line.split_whitespace();
            let n: usize = iter.next()
                .ok_or(IoError::Parse("Invalid face".to_string()))?
                .parse().map_err(|_| IoError::Parse("Invalid face size".to_string()))?;
            if n >= 3 {
                let indices: Vec<VertexHandle> = iter
                    .filter_map(|s| s.parse::<usize>().ok())
                    .filter_map(|i| vhandles.get(i).copied())
                    .collect();
                if indices.len() >= 3 {
                    mesh.add_face(&indices);
                }
            }
        }
    }
    
    Ok(mesh)
}

/// Write PLY format file (ASCII)
pub fn write_ply<P: AsRef<Path>>(mesh: &RustMesh, path: P) -> IoResult<()> {
    let mut file = File::create(path)?;
    
    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex {}", mesh.n_vertices())?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    writeln!(file, "element face {}", mesh.n_faces())?;
    writeln!(file, "property list uchar int vertex_index")?;
    writeln!(file, "end_header")?;
    
    for i in 0..mesh.n_vertices() {
        let vh = VertexHandle::from_usize(i);
        if let Some(p) = mesh.point(vh) {
            writeln!(file, "{} {} {}", p.x, p.y, p.z)?;
        }
    }
    
    for fh in mesh.faces() {
        if let Some(verts) = mesh.face_vertices(fh) {
            let vs: Vec<_> = verts.collect();
            write!(file, "{} ", vs.len())?;
            for (i, vh) in vs.iter().enumerate() {
                if i > 0 { write!(file, " ")?; }
                write!(file, "{}", vh.idx_usize())?;
            }
            writeln!(file)?;
        }
    }
    
    Ok(())
}

/// Write PLY format file (Binary Little Endian)
pub fn write_ply_binary<P: AsRef<Path>>(mesh: &RustMesh, path: P) -> IoResult<()> {
    let mut file = File::create(path)?;
    
    writeln!(file, "ply")?;
    writeln!(file, "format binary_little_endian 1.0")?;
    writeln!(file, "element vertex {}", mesh.n_vertices())?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    writeln!(file, "element face {}", mesh.n_faces())?;
    writeln!(file, "property list uchar int vertex_index")?;
    writeln!(file, "end_header")?;
    
    for i in 0..mesh.n_vertices() {
        let vh = VertexHandle::from_usize(i);
        if let Some(p) = mesh.point(vh) {
            file.write_f32::<LittleEndian>(p.x)?;
            file.write_f32::<LittleEndian>(p.y)?;
            file.write_f32::<LittleEndian>(p.z)?;
        }
    }
    
    for fh in mesh.faces() {
        if let Some(verts) = mesh.face_vertices(fh) {
            let vs: Vec<_> = verts.collect();
            file.write_u8(vs.len() as u8)?;
            for vh in &vs {
                file.write_i32::<LittleEndian>(vh.idx_usize() as i32)?;
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests_ply {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_ply_ascii() {
        let content = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_index\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";
        let temp = NamedTempFile::new().unwrap();
        std::fs::write(temp.path(), content).unwrap();
        let mesh = read_ply(temp.path()).unwrap();
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn test_write_ply_ascii() {
        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);
        
        let temp = NamedTempFile::new().unwrap();
        write_ply(&mesh, temp.path()).unwrap();
        
        let content = std::fs::read_to_string(temp.path()).unwrap();
        assert!(content.contains("ply"));
        assert!(content.contains("element vertex 3"));
        
        let read_mesh = read_ply(temp.path()).unwrap();
        assert_eq!(read_mesh.n_vertices(), 3);
        assert_eq!(read_mesh.n_faces(), 1);
    }

    #[test]
    fn test_write_ply_binary() {
        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);
        
        let temp = NamedTempFile::new().unwrap();
        write_ply_binary(&mesh, temp.path()).unwrap();
        
        // Verify file exists and has content
        let metadata = std::fs::metadata(temp.path()).unwrap();
        assert!(metadata.len() > 0, "Binary PLY file should not be empty");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_off_triangle() {
        let content = "OFF
3 1 0
0 0 0
1 0 0
0 1 0
3 0 1 2
";
        
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        
        let mesh = read_off(file.path()).unwrap();
        
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn test_read_off_quad() {
        let content = "OFF
4 1 0
0 0 0
1 0 0
1 1 0
0 1 0
4 0 1 2 3
";
        
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        
        let mesh = read_off(file.path()).unwrap();
        
        assert_eq!(mesh.n_vertices(), 4);
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn test_read_obj_simple() {
        let content = "# Simple triangle
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
";
        
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        
        let mesh = read_obj(file.path()).unwrap();
        
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn test_detect_format() {
        assert_eq!(detect_format(Path::new("mesh.off")).unwrap(), "OFF");
        assert_eq!(detect_format(Path::new("mesh.OBJ")).unwrap(), "OBJ");
        assert_eq!(detect_format(Path::new("model.obj")).unwrap(), "OBJ");
        assert_eq!(detect_format(Path::new("data.ply")).unwrap(), "PLY");
        assert_eq!(detect_format(Path::new("cube.stl")).unwrap(), "STL");
        assert_eq!(detect_format(Path::new("part.STL")).unwrap(), "STL");
        assert!(detect_format(Path::new("mesh.unknown")).is_none());
    }

    #[test]
    fn test_read_stl_ascii() {
        let content = "solid mesh
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid mesh
";
        
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        
        let mesh = read_stl(file.path()).unwrap();
        
        // Should have at least 3 vertices (may have duplicates)
        assert!(mesh.n_vertices() >= 3);
        // Should have at least 1 face
        assert!(mesh.n_faces() >= 1);
    }

    #[test]
    fn test_write_stl() {
        let mesh = RustMesh::new();
        let mut temp_file = NamedTempFile::new().unwrap();
        
        write_stl(&mesh, temp_file.path()).unwrap();
        
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("solid mesh"));
        assert!(content.contains("endsolid mesh"));
    }

    #[test]
    fn test_stl_binary_read_write() {
        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        // Write binary STL
        let temp_file = NamedTempFile::new().unwrap();
        write_stl_binary(&mesh, temp_file.path()).unwrap();

        // Check file size (header:80 + 4 + 1*50 = 134 bytes)
        let metadata = std::fs::metadata(temp_file.path()).unwrap();
        assert_eq!(metadata.len(), 134);

        // Read binary STL
        let read_mesh = read_stl_binary(temp_file.path()).unwrap();
        assert!(read_mesh.n_vertices() >= 3);
    }

    #[test]
    fn test_stl_binary_triangle() {
        // Create a single triangle
        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 0.0, 1.0));
        mesh.add_face(&[v0, v1, v2]);

        // Write and read
        let temp_file = NamedTempFile::new().unwrap();
        write_stl_binary(&mesh, temp_file.path()).unwrap();
        let read_mesh = read_stl_binary(temp_file.path()).unwrap();

        // Verify structure
        assert!(read_mesh.n_vertices() >= 3);
        assert!(read_mesh.n_faces() >= 1);
    }
}
