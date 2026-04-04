//! PLY File Format Support
//!
//! Stanford PLY (Polygon File Format) supports both ASCII and binary formats.
//!
//! Format specification:
//! - Header with element definitions
//! - Vertex list (x, y, z, nx, ny, nz, red, green, blue)
//! - Face list (vertex count + indices)

use crate::RustMesh;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
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

/// PLY property type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyPropertyType {
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Float,
    Double,
}

impl PlyPropertyType {
    /// Parse from string
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "char" | "int8" => Some(Self::Char),
            "uchar" | "uint8" => Some(Self::UChar),
            "short" | "int16" => Some(Self::Short),
            "ushort" | "uint16" => Some(Self::UShort),
            "int" | "int32" => Some(Self::Int),
            "uint" | "uint32" => Some(Self::UInt),
            "float" | "float32" => Some(Self::Float),
            "double" | "float64" => Some(Self::Double),
            _ => None,
        }
    }

    /// Size in bytes
    fn size(&self) -> usize {
        match self {
            Self::Char | Self::UChar => 1,
            Self::Short | Self::UShort => 2,
            Self::Int | Self::UInt | Self::Float => 4,
            Self::Double => 8,
        }
    }
}

/// PLY property definition
#[derive(Debug, Clone)]
struct PlyProperty {
    name: String,
    data_type: PlyPropertyType,
    is_list: bool,
    list_size_type: Option<PlyPropertyType>,
}

/// PLY element definition
#[derive(Debug, Clone)]
struct PlyElement {
    name: String,
    count: usize,
    properties: Vec<PlyProperty>,
}

/// PLY header information
#[derive(Debug)]
struct PlyHeader {
    format: PlyFormat,
    version: String,
    elements: Vec<PlyElement>,
    comments: Vec<String>,
}

// ============================================================================
// PLY Reading Implementation
// ============================================================================

/// Read mesh from PLY file
pub fn read_ply(path: impl AsRef<Path>) -> io::Result<RustMesh> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Parse header
    let (header, mut data_reader) = parse_ply_header(reader)?;

    // Read data based on format
    match header.format {
        PlyFormat::Ascii => read_ply_ascii_data(data_reader, &header),
        PlyFormat::BinaryLittleEndian => {
            // For binary, we need to read remaining buffered data first
            read_ply_binary_data_from_buffered(&mut data_reader, &header, false)
        }
        PlyFormat::BinaryBigEndian => {
            read_ply_binary_data_from_buffered(&mut data_reader, &header, true)
        }
    }
}

/// Parse PLY header
/// Returns the header info and the remaining reader for data
fn parse_ply_header(mut reader: BufReader<File>) -> io::Result<(PlyHeader, BufReader<File>)> {
    let mut line = String::new();

    // First line must be "ply"
    reader.read_line(&mut line)?;
    if !line.trim().eq_ignore_ascii_case("ply") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not a PLY file: missing 'ply' header",
        ));
    }

    let mut format = PlyFormat::Ascii;
    let mut version = String::new();
    let mut elements: Vec<PlyElement> = Vec::new();
    let mut comments: Vec<String> = Vec::new();
    let mut current_element: Option<PlyElement> = None;

    loop {
        line.clear();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0].to_lowercase().as_str() {
            "format" => {
                if parts.len() >= 3 {
                    format = match parts[1].to_lowercase().as_str() {
                        "ascii" => PlyFormat::Ascii,
                        "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                        "binary_big_endian" => PlyFormat::BinaryBigEndian,
                        _ => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("Unknown PLY format: {}", parts[1]),
                            ))
                        }
                    };
                    version = parts[2].to_string();
                }
            }
            "element" => {
                // Save previous element
                if let Some(elem) = current_element.take() {
                    elements.push(elem);
                }

                if parts.len() >= 3 {
                    current_element = Some(PlyElement {
                        name: parts[1].to_string(),
                        count: parts[2].parse().unwrap_or(0),
                        properties: Vec::new(),
                    });
                }
            }
            "property" => {
                if let Some(ref mut elem) = current_element {
                    if parts.len() >= 3 {
                        if parts[1].to_lowercase() == "list" {
                            // List property: "property list <size_type> <data_type> <name>"
                            if parts.len() >= 5 {
                                let list_size_type = PlyPropertyType::from_str(parts[2]);
                                let data_type = PlyPropertyType::from_str(parts[3]);
                                if let (Some(lst), Some(dt)) = (list_size_type, data_type) {
                                    elem.properties.push(PlyProperty {
                                        name: parts[4].to_string(),
                                        data_type: dt,
                                        is_list: true,
                                        list_size_type: Some(lst),
                                    });
                                }
                            }
                        } else {
                            // Regular property: "property <type> <name>"
                            if let Some(dt) = PlyPropertyType::from_str(parts[1]) {
                                elem.properties.push(PlyProperty {
                                    name: parts[2].to_string(),
                                    data_type: dt,
                                    is_list: false,
                                    list_size_type: None,
                                });
                            }
                        }
                    }
                }
            }
            "comment" => {
                comments.push(trimmed[7..].to_string());
            }
            "end_header" => {
                // Save last element
                if let Some(elem) = current_element.take() {
                    elements.push(elem);
                }
                break;
            }
            _ => {}
        }
    }

    Ok((
        PlyHeader {
            format,
            version,
            elements,
            comments,
        },
        reader,
    ))
}

/// Build vertex property index map
fn build_vertex_property_map(element: &PlyElement) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for (i, prop) in element.properties.iter().enumerate() {
        map.insert(prop.name.clone(), i);
    }
    map
}

/// Read ASCII PLY data
fn read_ply_ascii_data(mut reader: BufReader<File>, header: &PlyHeader) -> io::Result<RustMesh> {
    let mut mesh = RustMesh::new();

    // Find vertex and face elements
    let vertex_element = header
        .elements
        .iter()
        .find(|e| e.name.to_lowercase() == "vertex");
    let face_element = header
        .elements
        .iter()
        .find(|e| e.name.to_lowercase() == "face");

    // Build property maps
    let vertex_props: Option<HashMap<String, usize>> =
        vertex_element.as_ref().map(|e| build_vertex_property_map(e));

    // Request attributes if present
    let has_normals = vertex_props
        .as_ref()
        .map(|p| p.contains_key("nx") || p.contains_key("normal_x"))
        .unwrap_or(false);
    let has_colors = vertex_props
        .as_ref()
        .map(|p| p.contains_key("red") || p.contains_key("diffuse_red"))
        .unwrap_or(false);

    if has_normals {
        mesh.request_vertex_normals();
    }
    if has_colors {
        mesh.request_vertex_colors();
    }

    // Read vertices
    if let Some(ref elem) = vertex_element {
        let prop_map = vertex_props.as_ref().unwrap();

        for _ in 0..elem.count {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            // Parse position
            let x_idx = prop_map.get("x").or_else(|| prop_map.get("pos_x"));
            let y_idx = prop_map.get("y").or_else(|| prop_map.get("pos_y"));
            let z_idx = prop_map.get("z").or_else(|| prop_map.get("pos_z"));

            let x = x_idx
                .and_then(|&i| parts.get(i).and_then(|s| s.parse::<f32>().ok()))
                .unwrap_or(0.0);
            let y = y_idx
                .and_then(|&i| parts.get(i).and_then(|s| s.parse::<f32>().ok()))
                .unwrap_or(0.0);
            let z = z_idx
                .and_then(|&i| parts.get(i).and_then(|s| s.parse::<f32>().ok()))
                .unwrap_or(0.0);

            let vh = mesh.add_vertex(glam::Vec3::new(x, y, z));

            // Parse normal
            if has_normals {
                let nx_idx = prop_map.get("nx").or_else(|| prop_map.get("normal_x"));
                let ny_idx = prop_map.get("ny").or_else(|| prop_map.get("normal_y"));
                let nz_idx = prop_map.get("nz").or_else(|| prop_map.get("normal_z"));

                let nx = nx_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<f32>().ok()))
                    .unwrap_or(0.0);
                let ny = ny_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<f32>().ok()))
                    .unwrap_or(0.0);
                let nz = nz_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<f32>().ok()))
                    .unwrap_or(1.0);

                mesh.set_vertex_normal_by_index(vh.idx_usize(), glam::Vec3::new(nx, ny, nz));
            }

            // Parse color
            if has_colors {
                let r_idx = prop_map.get("red").or_else(|| prop_map.get("diffuse_red"));
                let g_idx = prop_map.get("green").or_else(|| prop_map.get("diffuse_green"));
                let b_idx = prop_map.get("blue").or_else(|| prop_map.get("diffuse_blue"));
                let a_idx = prop_map.get("alpha").or_else(|| prop_map.get("diffuse_alpha"));

                let r = r_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<u8>().ok()))
                    .unwrap_or(255);
                let g = g_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<u8>().ok()))
                    .unwrap_or(255);
                let b = b_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<u8>().ok()))
                    .unwrap_or(255);
                let a = a_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<u8>().ok()))
                    .unwrap_or(255);

                mesh.set_vertex_color_by_index(
                    vh.idx_usize(),
                    glam::Vec4::new(
                        r as f32 / 255.0,
                        g as f32 / 255.0,
                        b as f32 / 255.0,
                        a as f32 / 255.0,
                    ),
                );
            }
        }
    }

    // Read faces
    if let Some(ref elem) = face_element {
        for _ in 0..elem.count {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.is_empty() {
                continue;
            }

            // First value is the vertex count for this face
            let n_verts: usize = parts[0].parse().unwrap_or(0);
            if n_verts < 3 || n_verts > parts.len() - 1 {
                continue;
            }

            let mut indices = Vec::with_capacity(n_verts);
            for i in 1..=n_verts {
                if let Ok(idx) = parts[i].parse::<u32>() {
                    indices.push(crate::VertexHandle::new(idx));
                }
            }

            if indices.len() >= 3 {
                mesh.add_face(&indices);
            }
        }
    }

    Ok(mesh)
}

/// Read binary PLY data from buffered reader
fn read_ply_binary_data_from_buffered(
    reader: &mut BufReader<File>,
    header: &PlyHeader,
    big_endian: bool,
) -> io::Result<RustMesh> {
    let mut mesh = RustMesh::new();

    // Find vertex and face elements
    let vertex_element = header
        .elements
        .iter()
        .find(|e| e.name.to_lowercase() == "vertex");
    let face_element = header
        .elements
        .iter()
        .find(|e| e.name.to_lowercase() == "face");

    // Build property maps
    let vertex_props: Option<HashMap<String, usize>> =
        vertex_element.as_ref().map(|e| build_vertex_property_map(e));

    // Request attributes if present
    let has_normals = vertex_props
        .as_ref()
        .map(|p| p.contains_key("nx") || p.contains_key("normal_x"))
        .unwrap_or(false);
    let has_colors = vertex_props
        .as_ref()
        .map(|p| p.contains_key("red") || p.contains_key("diffuse_red"))
        .unwrap_or(false);

    if has_normals {
        mesh.request_vertex_normals();
    }
    if has_colors {
        mesh.request_vertex_colors();
    }

    // Helper function to read a binary value from BufReader
    fn read_value_buf(reader: &mut BufReader<File>, ty: PlyPropertyType, big_endian: bool) -> io::Result<f64> {
        Ok(match ty {
            PlyPropertyType::Char => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                buf[0] as i8 as f64
            }
            PlyPropertyType::UChar => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                buf[0] as f64
            }
            PlyPropertyType::Short => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf)?;
                let val = if big_endian {
                    i16::from_be_bytes(buf)
                } else {
                    i16::from_le_bytes(buf)
                };
                val as f64
            }
            PlyPropertyType::UShort => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf)?;
                let val = if big_endian {
                    u16::from_be_bytes(buf)
                } else {
                    u16::from_le_bytes(buf)
                };
                val as f64
            }
            PlyPropertyType::Int => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                let val = if big_endian {
                    i32::from_be_bytes(buf)
                } else {
                    i32::from_le_bytes(buf)
                };
                val as f64
            }
            PlyPropertyType::UInt => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                let val = if big_endian {
                    u32::from_be_bytes(buf)
                } else {
                    u32::from_le_bytes(buf)
                };
                val as f64
            }
            PlyPropertyType::Float => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                let val = if big_endian {
                    f32::from_be_bytes(buf)
                } else {
                    f32::from_le_bytes(buf)
                };
                val as f64
            }
            PlyPropertyType::Double => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                if big_endian {
                    f64::from_be_bytes(buf)
                } else {
                    f64::from_le_bytes(buf)
                }
            }
        })
    }

    // Read vertices
    if let Some(ref elem) = vertex_element {
        let prop_map = vertex_props.as_ref().unwrap();

        for _ in 0..elem.count {
            let mut values = Vec::with_capacity(elem.properties.len());

            for prop in &elem.properties {
                let val = read_value_buf(reader, prop.data_type, big_endian)?;
                values.push(val);
            }

            // Parse position
            let x_idx = prop_map.get("x").or_else(|| prop_map.get("pos_x"));
            let y_idx = prop_map.get("y").or_else(|| prop_map.get("pos_y"));
            let z_idx = prop_map.get("z").or_else(|| prop_map.get("pos_z"));

            let x = x_idx.and_then(|&i| values.get(i)).copied().unwrap_or(0.0) as f32;
            let y = y_idx.and_then(|&i| values.get(i)).copied().unwrap_or(0.0) as f32;
            let z = z_idx.and_then(|&i| values.get(i)).copied().unwrap_or(0.0) as f32;

            let vh = mesh.add_vertex(glam::Vec3::new(x, y, z));

            // Parse normal
            if has_normals {
                let nx_idx = prop_map.get("nx").or_else(|| prop_map.get("normal_x"));
                let ny_idx = prop_map.get("ny").or_else(|| prop_map.get("normal_y"));
                let nz_idx = prop_map.get("nz").or_else(|| prop_map.get("normal_z"));

                let nx = nx_idx.and_then(|&i| values.get(i)).copied().unwrap_or(0.0) as f32;
                let ny = ny_idx.and_then(|&i| values.get(i)).copied().unwrap_or(0.0) as f32;
                let nz = nz_idx.and_then(|&i| values.get(i)).copied().unwrap_or(1.0) as f32;

                mesh.set_vertex_normal_by_index(vh.idx_usize(), glam::Vec3::new(nx, ny, nz));
            }

            // Parse color
            if has_colors {
                let r_idx = prop_map.get("red").or_else(|| prop_map.get("diffuse_red"));
                let g_idx = prop_map.get("green").or_else(|| prop_map.get("diffuse_green"));
                let b_idx = prop_map.get("blue").or_else(|| prop_map.get("diffuse_blue"));
                let a_idx = prop_map.get("alpha").or_else(|| prop_map.get("diffuse_alpha"));

                let r = r_idx.and_then(|&i| values.get(i)).copied().unwrap_or(255.0) as f32 / 255.0;
                let g = g_idx.and_then(|&i| values.get(i)).copied().unwrap_or(255.0) as f32 / 255.0;
                let b = b_idx.and_then(|&i| values.get(i)).copied().unwrap_or(255.0) as f32 / 255.0;
                let a = a_idx.and_then(|&i| values.get(i)).copied().unwrap_or(255.0) as f32 / 255.0;

                mesh.set_vertex_color_by_index(vh.idx_usize(), glam::Vec4::new(r, g, b, a));
            }
        }
    }

    // Read faces
    if let Some(ref elem) = face_element {
        for _ in 0..elem.count {
            // For faces, we need to handle list properties
            // First read the list size
            let list_prop = elem.properties.iter().find(|p| p.is_list);
            if let Some(prop) = list_prop {
                // Read list size
                let size_type = prop.list_size_type.unwrap_or(PlyPropertyType::UChar);
                let n_verts = read_value_buf(reader, size_type, big_endian)? as usize;

                // Read indices
                let mut indices = Vec::with_capacity(n_verts);
                for _ in 0..n_verts {
                    let idx = read_value_buf(reader, prop.data_type, big_endian)? as u32;
                    indices.push(crate::VertexHandle::new(idx));
                }

                if indices.len() >= 3 {
                    mesh.add_face(&indices);
                }
            }
        }
    }

    Ok(mesh)
}

// ============================================================================
// PLY Writing Implementation
// ============================================================================

/// Write mesh to PLY file
pub fn write_ply(mesh: &RustMesh, path: impl AsRef<Path>, format: PlyFormat) -> io::Result<()> {
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
fn write_ply_binary(mesh: &RustMesh, path: impl AsRef<Path>, big_endian: bool) -> io::Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mesh() -> RustMesh {
        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(glam::Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::Vec3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);
        mesh
    }

    #[test]
    fn test_ply_ascii_write() {
        use std::fs;

        let mesh = create_test_mesh();
        let path = "/tmp/test_mesh.ply";
        write_ply(&mesh, path, PlyFormat::Ascii).unwrap();

        // Verify file exists
        assert!(std::path::Path::new(path).exists());

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_ply_binary_write() {
        use std::fs;

        let mesh = create_test_mesh();
        let path = "/tmp/test_mesh_binary.ply";
        write_ply(&mesh, path, PlyFormat::BinaryLittleEndian).unwrap();

        assert!(std::path::Path::new(path).exists());

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_ply_ascii_roundtrip() {
        use std::fs;

        let original = create_test_mesh();
        let path = "/tmp/test_roundtrip.ply";

        // Write
        write_ply(&original, path, PlyFormat::Ascii).unwrap();

        // Read
        let loaded = read_ply(path).unwrap();

        // Verify
        assert_eq!(loaded.n_vertices(), original.n_vertices());
        assert_eq!(loaded.n_faces(), original.n_faces());

        // Clean up
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_ply_binary_roundtrip() {
        use std::fs;

        let original = create_test_mesh();
        let path = "/tmp/test_roundtrip_binary.ply";

        // Write
        write_ply(&original, path, PlyFormat::BinaryLittleEndian).unwrap();

        // Read
        let loaded = read_ply(path).unwrap();

        // Verify
        assert_eq!(loaded.n_vertices(), original.n_vertices());
        assert_eq!(loaded.n_faces(), original.n_faces());

        // Clean up
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_ply_with_normals() {
        use std::fs;

        let mut mesh = RustMesh::new();
        mesh.request_vertex_normals();

        let v0 = mesh.add_vertex(glam::Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::Vec3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        mesh.set_vertex_normal_by_index(0, glam::Vec3::new(0.0, 0.0, 1.0));
        mesh.set_vertex_normal_by_index(1, glam::Vec3::new(0.0, 0.0, 1.0));
        mesh.set_vertex_normal_by_index(2, glam::Vec3::new(0.0, 0.0, 1.0));

        let path = "/tmp/test_normals.ply";
        write_ply(&mesh, path, PlyFormat::Ascii).unwrap();

        let loaded = read_ply(path).unwrap();

        assert_eq!(loaded.n_vertices(), 3);
        assert!(loaded.has_vertex_normals());

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_ply_with_colors() {
        use std::fs;

        let mut mesh = RustMesh::new();
        mesh.request_vertex_colors();

        let v0 = mesh.add_vertex(glam::Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::Vec3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        mesh.set_vertex_color_by_index(0, glam::Vec4::new(1.0, 0.0, 0.0, 1.0));
        mesh.set_vertex_color_by_index(1, glam::Vec4::new(0.0, 1.0, 0.0, 1.0));
        mesh.set_vertex_color_by_index(2, glam::Vec4::new(0.0, 0.0, 1.0, 1.0));

        let path = "/tmp/test_colors.ply";
        write_ply(&mesh, path, PlyFormat::Ascii).unwrap();

        let loaded = read_ply(path).unwrap();

        assert_eq!(loaded.n_vertices(), 3);
        assert!(loaded.has_vertex_colors());

        fs::remove_file(path).ok();
    }
}
