//! STL File Format Support
//!
//! STereoLithography format for 3D printing applications.

use crate::RustMesh;
use std::io;
use std::path::Path;

/// STL file format variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StlFormat {
    /// ASCII text format
    Ascii,
    /// Binary format
    Binary,
}

/// Write mesh to STL file
pub fn write_stl(_mesh: &RustMesh, _path: impl AsRef<Path>, _format: StlFormat) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "STL writing not yet implemented",
    ))
}

/// Read mesh from STL file
pub fn read_stl(_path: impl AsRef<Path>) -> io::Result<RustMesh> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "STL reading not yet implemented",
    ))
}
