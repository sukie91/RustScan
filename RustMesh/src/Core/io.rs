//! # IO Module - File Import/Export
//!
//! Supports reading and writing mesh files in various formats:
//! - OBJ (Wavefront)
//! - PLY (Polygon File Format)
//! - STL (STereoLithography)
//! - OFF (Object File Format)

pub mod obj;
pub mod ply;
pub mod stl;
pub mod off;

// Re-export for convenience
pub use obj::{read_obj, write_obj};
pub use ply::{read_ply, write_ply, PlyFormat};
pub use stl::{read_stl, write_stl, StlFormat};
pub use off::{read_off, write_off};

use std::path::Path;

/// Detect file format from extension
pub fn detect_format(path: impl AsRef<Path>) -> Option<&'static str> {
    let path = path.as_ref();
    let ext = path.extension()?.to_str()?.to_lowercase();

    match ext.as_str() {
        "obj" => Some("OBJ"),
        "ply" => Some("PLY"),
        "stl" => Some("STL"),
        "off" => Some("OFF"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format() {
        assert_eq!(detect_format("mesh.obj"), Some("OBJ"));
        assert_eq!(detect_format("mesh.ply"), Some("PLY"));
        assert_eq!(detect_format("mesh.stl"), Some("STL"));
        assert_eq!(detect_format("mesh.off"), Some("OFF"));
        assert_eq!(detect_format("mesh.unknown"), None);
    }
}
