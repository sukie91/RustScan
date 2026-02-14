//! OFF File Format Support
//!
//! Object File Format - simple format for polygonal meshes.

use crate::RustMesh;
use std::io;
use std::path::Path;

/// Write mesh to OFF file
pub fn write_off(_mesh: &RustMesh, _path: impl AsRef<Path>) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "OFF writing not yet implemented",
    ))
}

/// Read mesh from OFF file
pub fn read_off(_path: impl AsRef<Path>) -> io::Result<RustMesh> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "OFF reading not yet implemented",
    ))
}
