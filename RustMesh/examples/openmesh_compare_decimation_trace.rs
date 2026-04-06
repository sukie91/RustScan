mod openmesh_compare_common;

use openmesh_compare_common::{
    cleanup_paths, measure, mesh_digest, openmesh_root, print_duration_compare, print_header,
    print_mesh_digest, write_temp_off,
};
use rustmesh::{
    generate_sphere, read_off, read_off_openmesh_parity, Decimater, DecimationTraceStep,
    FaceHandle, VertexHandle,
};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
struct TraceStepRow {
    step: usize,
    removed: usize,
    kept: usize,
    boundary: bool,
    faces_removed: usize,
    priority: f64,
    active_faces_before: usize,
    active_faces_after: usize,
}

#[derive(Debug, Clone, Default)]
struct TraceRun {
    collapsed: usize,
    boundary_collapses: usize,
    interior_collapses: usize,
    final_vertices: usize,
    final_faces: usize,
    steps: Vec<TraceStepRow>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustImportMode {
    Standard,
    OpenMeshParity,
}

fn main() {
    let trace_limit = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(40);

    print_header("RustMesh vs OpenMesh Decimation Trace");
    println!("Reference sources:");
    println!("  - Mirror/OpenMesh-11.0.0/src/OpenMesh/Tools/Decimater/DecimaterT_impl.hh");
    println!("  - Mirror/OpenMesh-11.0.0/src/OpenMesh/Tools/Decimater/BaseDecimaterT_impl.hh");
    println!("  - Mirror/OpenMesh-11.0.0/src/OpenMesh/Tools/Decimater/ModQuadricT.hh");

    let input = generate_sphere(1.0, 10, 10);
    let input_path = write_temp_off(&input, "decimate-trace-input").expect("write input OFF");
    let target_vertices = input.n_vertices() / 2;

    print_mesh_digest("Input mesh", mesh_digest(&input));
    println!(
        "Trace configuration: target_vertices={}, trace_limit={}",
        target_vertices, trace_limit
    );
    let rust_import_mode = parse_rust_import_mode();
    if rust_import_mode != RustImportMode::Standard {
        println!("RustMesh import mode: {:?}", rust_import_mode);
    }

    let (rust_time, rust_trace) =
        measure(|| run_rust_trace(&input_path, target_vertices, trace_limit, rust_import_mode));
    let rust_trace = rust_trace.expect("run RustMesh trace");
    print_summary("RustMesh", &rust_trace);

    let (openmesh_time, openmesh_trace) =
        measure(|| run_openmesh_trace(&input_path, target_vertices, trace_limit));
    let openmesh_trace = openmesh_trace.expect("run OpenMesh trace");
    print_summary("OpenMesh", &openmesh_trace);

    print_duration_compare("Trace pipeline", rust_time, Some(openmesh_time));

    print_header("Trace Comparison");
    let matching_prefix = rust_trace
        .steps
        .iter()
        .zip(openmesh_trace.steps.iter())
        .take_while(|(lhs, rhs)| same_signature(lhs, rhs))
        .count();
    println!(
        "Matching prefix on removed/kept/boundary/faces_removed: {} steps",
        matching_prefix
    );
    let matching_edge_prefix = rust_trace
        .steps
        .iter()
        .zip(openmesh_trace.steps.iter())
        .take_while(|(lhs, rhs)| same_undirected_edge(lhs, rhs))
        .count();
    println!(
        "Matching prefix on undirected edge + faces_removed: {} steps",
        matching_edge_prefix
    );
    if let Some((lhs, rhs)) = rust_trace
        .steps
        .iter()
        .zip(openmesh_trace.steps.iter())
        .find(|(lhs, rhs)| !same_signature(lhs, rhs))
    {
        println!(
            "First divergence at step {}: RustMesh {} vs OpenMesh {}",
            lhs.step,
            format_step(lhs),
            format_step(rhs)
        );
    }

    let rust_boundary_prefix = rust_trace.steps.iter().filter(|step| step.boundary).count();
    let openmesh_boundary_prefix = openmesh_trace
        .steps
        .iter()
        .filter(|step| step.boundary)
        .count();
    println!(
        "Boundary collapses in first {} traced steps: RustMesh={}, OpenMesh={}",
        trace_limit, rust_boundary_prefix, openmesh_boundary_prefix
    );
    if let Some(step) = first_boundary_mix_gap(&rust_trace.steps, &openmesh_trace.steps) {
        println!(
            "First cumulative boundary/interior mix gap within trace: step {}",
            step
        );
    }

    if let Some((lhs, rhs)) = rust_trace
        .steps
        .iter()
        .zip(openmesh_trace.steps.iter())
        .find(|(lhs, rhs)| lhs.active_faces_after != rhs.active_faces_after)
    {
        println!(
            "First active-face gap at step {}: RustMesh={} OpenMesh={}",
            lhs.step, lhs.active_faces_after, rhs.active_faces_after
        );
    } else {
        println!("No active-face gap inside the traced prefix.");
    }

    println!("\nStep-by-step trace:");
    for idx in 0..trace_limit
        .min(rust_trace.steps.len())
        .min(openmesh_trace.steps.len())
    {
        let rust_step = &rust_trace.steps[idx];
        let openmesh_step = &openmesh_trace.steps[idx];
        println!(
            "#{:02} RustMesh: {:<44} | OpenMesh: {:<44} | {}",
            rust_step.step,
            format_step(rust_step),
            format_step(openmesh_step),
            if same_signature(rust_step, openmesh_step) {
                "match"
            } else {
                "diff"
            }
        );
    }

    cleanup_paths(&[input_path.as_path()]);
}

fn parse_rust_import_mode() -> RustImportMode {
    match std::env::var("RUSTMESH_TRACE_IMPORT_MODE").ok().as_deref() {
        Some("standard") => RustImportMode::Standard,
        Some("openmesh_parity") => RustImportMode::OpenMeshParity,
        _ => RustImportMode::OpenMeshParity,
    }
}

fn load_rust_trace_mesh(input_path: &Path, mode: RustImportMode) -> io::Result<rustmesh::RustMesh> {
    match mode {
        RustImportMode::Standard => read_off(input_path),
        RustImportMode::OpenMeshParity => read_off_openmesh_parity(input_path),
    }
}

fn run_rust_trace(
    input_path: &Path,
    target_vertices: usize,
    trace_limit: usize,
    mode: RustImportMode,
) -> io::Result<TraceRun> {
    let mut mesh = load_rust_trace_mesh(input_path, mode)?;
    let debug_vertices = parse_debug_vertices();
    if !debug_vertices.is_empty() {
        print_header("RustMesh Candidate Dump");
        debug_rust_vertices(&mut mesh, &debug_vertices, parse_debug_collapses_before());
        debug_rust_face_quadrics(&mesh, &debug_vertices);
    }
    let mut decimater = Decimater::new(&mut mesh);
    let trace = decimater.decimate_to_with_trace(target_vertices, trace_limit);
    let boundary_collapses = decimater.boundary_collapses();
    let interior_collapses = decimater.interior_collapses();
    drop(decimater);
    mesh.garbage_collection();
    let digest = mesh_digest(&mesh);

    Ok(TraceRun {
        collapsed: trace.collapsed,
        boundary_collapses,
        interior_collapses,
        final_vertices: digest.vertices,
        final_faces: digest.faces,
        steps: trace.steps.into_iter().map(step_from_rust).collect(),
    })
}

fn parse_debug_vertices() -> Vec<usize> {
    std::env::var("RUSTMESH_TRACE_DUMP_VERTICES")
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|token| token.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn parse_debug_collapses_before() -> usize {
    std::env::var("RUSTMESH_TRACE_DUMP_AFTER_COLLAPSES")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(0)
}

fn debug_rust_vertices(mesh: &mut rustmesh::RustMesh, vertices: &[usize], collapses_before: usize) {
    let mut decimater = Decimater::new(mesh);
    if collapses_before > 0 {
        let _ = decimater.decimate_with_trace(collapses_before, 0);
        println!("after {} collapses:", collapses_before);
    } else {
        decimater.debug_prepare_state();
    }

    for idx in vertices.iter().copied() {
        let vh = VertexHandle::from_usize(idx);
        let state = decimater.debug_vertex_state(vh);

        if !state.exists || state.is_deleted {
            println!("vertex {}: invalid", idx);
            continue;
        }

        println!(
            "vertex {}: anchor={:?} boundary_vertex={} point={:?} stored_in_heap={} heap_target={:?} heap_priority={:?}",
            idx,
            state.anchor,
            state.is_boundary_vertex,
            state.point,
            state.stored_in_heap,
            state.heap_target,
            state.heap_priority
        );
        if debug_dump_exact_bits_enabled() {
            if let Some(point) = state.point {
                println!(
                    "  point_bits=[0x{:08x},0x{:08x},0x{:08x}]",
                    point.x.to_bits(),
                    point.y.to_bits(),
                    point.z.to_bits()
                );
            }
        }
        if let Some(q) = &state.quadric {
            println!(
                "  quadric=[{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}]",
                q.a, q.b, q.c, q.d, q.e, q.f, q.g, q.h, q.i, q.j
            );
            if debug_dump_exact_bits_enabled() {
                println!(
                    "  quadric_bits=[0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x}]",
                    q.a.to_bits(),
                    q.b.to_bits(),
                    q.c.to_bits(),
                    q.d.to_bits(),
                    q.e.to_bits(),
                    q.f.to_bits(),
                    q.g.to_bits(),
                    q.h.to_bits(),
                    q.i.to_bits(),
                    q.j.to_bits()
                );
            }
        }

        if state.outgoing.is_empty() {
            println!("  no outgoing halfedges");
            continue;
        }

        for row in state.outgoing {
            match row.priority {
                Some(priority) => println!(
                    "  heh={:?} {}->{} boundary={} legal={} raw_error={:.12e} error={:.12e}",
                    row.halfedge,
                    idx,
                    row.v_to.idx_usize(),
                    row.is_boundary,
                    row.is_legal,
                    row.raw_error.unwrap_or(priority as f64),
                    priority
                ),
                None => println!(
                    "  heh={:?} {}->{} boundary={} legal={} raw_error={} illegal",
                    row.halfedge,
                    idx,
                    row.v_to.idx_usize(),
                    row.is_boundary,
                    row.is_legal,
                    row.raw_error
                        .map(|value| format!("{value:.12e}"))
                        .unwrap_or_else(|| "none".to_string())
                ),
            }
        }
    }
}

fn debug_dump_exact_bits_enabled() -> bool {
    std::env::var("RUSTMESH_TRACE_DUMP_EXACT")
        .ok()
        .is_some_and(|raw| !raw.trim().is_empty() && raw != "0")
}

fn debug_dump_face_quadrics_enabled() -> bool {
    std::env::var("RUSTMESH_TRACE_DUMP_FACE_QUADRICS")
        .ok()
        .is_some_and(|raw| !raw.trim().is_empty() && raw != "0")
}

fn parse_debug_face_ops() -> Vec<usize> {
    std::env::var("RUSTMESH_TRACE_DUMP_FACE_OPS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|token| token.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn should_debug_face_ops(face_idx: usize, debug_faces: &[usize]) -> bool {
    debug_faces.contains(&face_idx)
}

fn debug_rust_face_quadrics(mesh: &rustmesh::RustMesh, vertices: &[usize]) {
    if !debug_dump_face_quadrics_enabled() || vertices.is_empty() {
        return;
    }

    let debug_face_ops = parse_debug_face_ops();

    let mut faces = vertices
        .iter()
        .copied()
        .flat_map(|idx| {
            let vh = VertexHandle::from_usize(idx);
            mesh.vertex_faces(vh)
                .map(|iter| iter.map(|fh| fh.idx_usize()).collect::<Vec<_>>())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    faces.sort_unstable();
    faces.dedup();

    if faces.is_empty() {
        return;
    }

    print_header("RustMesh Face Quadric Dump");
    for face_idx in faces {
        let fh = FaceHandle::from_usize(face_idx);
        let verts_vh = mesh.face_vertices_vec(fh);
        let verts = verts_vh.iter().map(|vh| vh.idx_usize()).collect::<Vec<_>>();
        println!("face {}: verts={verts:?}", face_idx);
        if let Some(bits) = rust_face_quadric_bits_from_vertices(mesh, &verts_vh) {
            println!(
                "  face_quadric_bits=[0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x}]",
                bits[0], bits[1], bits[2], bits[3], bits[4], bits[5], bits[6], bits[7], bits[8], bits[9]
            );
            if should_debug_face_ops(face_idx, &debug_face_ops) {
                if let Some(ops) = rust_face_quadric_ops_from_vertices(mesh, &verts_vh) {
                    print_face_ops("  face_ops", &ops);
                }
            }
            for shift in 1..verts_vh.len().min(3) {
                let rotated = rotate_face_vertices(&verts_vh, shift);
                if let Some(rot_bits) = rust_face_quadric_bits_from_vertices(mesh, &rotated) {
                    println!(
                        "  face_quadric_bits_rot{shift}=[0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x}]",
                        rot_bits[0],
                        rot_bits[1],
                        rot_bits[2],
                        rot_bits[3],
                        rot_bits[4],
                        rot_bits[5],
                        rot_bits[6],
                        rot_bits[7],
                        rot_bits[8],
                        rot_bits[9]
                    );
                }
                if should_debug_face_ops(face_idx, &debug_face_ops) {
                    if let Some(rot_ops) = rust_face_quadric_ops_from_vertices(mesh, &rotated) {
                        print_face_ops(&format!("  face_ops_rot{shift}"), &rot_ops);
                    }
                }
            }
        } else {
            println!("  face_quadric_bits=none");
        }
    }
}

fn rust_face_quadric_bits(mesh: &rustmesh::RustMesh, fh: FaceHandle) -> Option<[u64; 10]> {
    let verts = mesh.face_vertices_vec(fh);
    rust_face_quadric_bits_from_vertices(mesh, &verts)
}

fn rust_face_quadric_bits_from_vertices(
    mesh: &rustmesh::RustMesh,
    verts: &[VertexHandle],
) -> Option<[u64; 10]> {
    if verts.len() < 3 {
        return None;
    }

    let p0 = mesh.point(verts[0])?;
    let p1 = mesh.point(verts[1])?;
    let p2 = mesh.point(verts[2])?;
    rust_face_quadric_bits_from_points(p0, p1, p2)
}

fn rotate_face_vertices(vertices: &[VertexHandle], shift: usize) -> Vec<VertexHandle> {
    if vertices.is_empty() {
        return Vec::new();
    }

    let n = vertices.len();
    (0..n).map(|idx| vertices[(idx + shift) % n]).collect()
}

#[derive(Clone, Copy)]
struct FaceQuadricOps {
    e1: [u64; 3],
    e2: [u64; 3],
    cross: [u64; 3],
    sqrnorm: u64,
    area_norm: u64,
    unit_normal: [u64; 3],
    area_half: u64,
    plane_dot: u64,
    d: u64,
}

fn print_face_ops(label: &str, ops: &FaceQuadricOps) {
    println!(
        "{label} e1=[0x{:016x},0x{:016x},0x{:016x}] e2=[0x{:016x},0x{:016x},0x{:016x}] cross=[0x{:016x},0x{:016x},0x{:016x}] sqrnorm=0x{:016x} area_norm=0x{:016x} unit_normal=[0x{:016x},0x{:016x},0x{:016x}] area_half=0x{:016x} plane_dot=0x{:016x} d=0x{:016x}",
        ops.e1[0],
        ops.e1[1],
        ops.e1[2],
        ops.e2[0],
        ops.e2[1],
        ops.e2[2],
        ops.cross[0],
        ops.cross[1],
        ops.cross[2],
        ops.sqrnorm,
        ops.area_norm,
        ops.unit_normal[0],
        ops.unit_normal[1],
        ops.unit_normal[2],
        ops.area_half,
        ops.plane_dot,
        ops.d
    );
}

fn rust_face_quadric_ops_from_vertices(
    mesh: &rustmesh::RustMesh,
    verts: &[VertexHandle],
) -> Option<FaceQuadricOps> {
    if verts.len() < 3 {
        return None;
    }

    let p0 = mesh.point(verts[0])?;
    let p1 = mesh.point(verts[1])?;
    let p2 = mesh.point(verts[2])?;
    rust_face_quadric_ops_from_points(p0, p1, p2)
}

fn rust_face_quadric_ops_from_points(
    p0: glam::Vec3,
    p1: glam::Vec3,
    p2: glam::Vec3,
) -> Option<FaceQuadricOps> {
    let p0x = p0.x as f64;
    let p0y = p0.y as f64;
    let p0z = p0.z as f64;
    let e1x = p1.x as f64 - p0x;
    let e1y = p1.y as f64 - p0y;
    let e1z = p1.z as f64 - p0z;
    let e2x = p2.x as f64 - p0x;
    let e2y = p2.y as f64 - p0y;
    let e2z = p2.z as f64 - p0z;

    let mut nx = e1y.mul_add(e2z, -(e1z * e2y));
    let mut ny = e1z.mul_add(e2x, -(e1x * e2z));
    let mut nz = e1x.mul_add(e2y, -(e1y * e2x));
    let cross = [nx.to_bits(), ny.to_bits(), nz.to_bits()];

    let mut sqrnorm = 0.0f64;
    sqrnorm = nx.mul_add(nx, sqrnorm);
    sqrnorm = ny.mul_add(ny, sqrnorm);
    sqrnorm = nz.mul_add(nz, sqrnorm);
    let area_norm = sqrnorm.sqrt();
    if !area_norm.is_finite() {
        return None;
    }

    let mut area_half = area_norm;
    if area_half > f32::MIN_POSITIVE as f64 {
        nx /= area_half;
        ny /= area_half;
        nz /= area_half;
        area_half *= 0.5;
    }

    let mut plane_dot = 0.0f64;
    plane_dot = p0x.mul_add(nx, plane_dot);
    plane_dot = p0y.mul_add(ny, plane_dot);
    plane_dot = p0z.mul_add(nz, plane_dot);
    let d = -plane_dot;

    Some(FaceQuadricOps {
        e1: [e1x.to_bits(), e1y.to_bits(), e1z.to_bits()],
        e2: [e2x.to_bits(), e2y.to_bits(), e2z.to_bits()],
        cross,
        sqrnorm: sqrnorm.to_bits(),
        area_norm: area_norm.to_bits(),
        unit_normal: [nx.to_bits(), ny.to_bits(), nz.to_bits()],
        area_half: area_half.to_bits(),
        plane_dot: plane_dot.to_bits(),
        d: d.to_bits(),
    })
}

fn rust_face_quadric_bits_from_points(
    p0: glam::Vec3,
    p1: glam::Vec3,
    p2: glam::Vec3,
) -> Option<[u64; 10]> {
    let p0x = p0.x as f64;
    let p0y = p0.y as f64;
    let p0z = p0.z as f64;
    let e1x = p1.x as f64 - p0x;
    let e1y = p1.y as f64 - p0y;
    let e1z = p1.z as f64 - p0z;
    let e2x = p2.x as f64 - p0x;
    let e2y = p2.y as f64 - p0y;
    let e2z = p2.z as f64 - p0z;

    let mut nx = e1y.mul_add(e2z, -(e1z * e2y));
    let mut ny = e1z.mul_add(e2x, -(e1x * e2z));
    let mut nz = e1x.mul_add(e2y, -(e1y * e2x));
    let mut sqrnorm = 0.0f64;
    sqrnorm = nx.mul_add(nx, sqrnorm);
    sqrnorm = ny.mul_add(ny, sqrnorm);
    sqrnorm = nz.mul_add(nz, sqrnorm);
    let mut area = sqrnorm.sqrt();
    if !area.is_finite() {
        return None;
    }

    if area > f32::MIN_POSITIVE as f64 {
        nx /= area;
        ny /= area;
        nz /= area;
        area *= 0.5;
    }

    let mut plane_dot = 0.0f64;
    plane_dot = p0x.mul_add(nx, plane_dot);
    plane_dot = p0y.mul_add(ny, plane_dot);
    plane_dot = p0z.mul_add(nz, plane_dot);
    let d = -plane_dot;
    let a = (nx * nx * area).to_bits();
    let b = (nx * ny * area).to_bits();
    let c = (nx * nz * area).to_bits();
    let d_bits = (nx * d * area).to_bits();
    let e = (ny * ny * area).to_bits();
    let f = (ny * nz * area).to_bits();
    let g = (ny * d * area).to_bits();
    let h = (nz * nz * area).to_bits();
    let i = (nz * d * area).to_bits();
    let j = (d * d * area).to_bits();
    Some([a, b, c, d_bits, e, f, g, h, i, j])
}

fn step_from_rust(step: DecimationTraceStep) -> TraceStepRow {
    TraceStepRow {
        step: step.step,
        removed: step.v_removed.idx_usize(),
        kept: step.v_kept.idx_usize(),
        boundary: step.is_boundary,
        faces_removed: step.faces_removed as usize,
        priority: step.priority as f64,
        active_faces_before: step.active_faces_before,
        active_faces_after: step.active_faces_after,
    }
}

fn run_openmesh_trace(
    input_path: &Path,
    target_vertices: usize,
    trace_limit: usize,
) -> io::Result<TraceRun> {
    let cpp_path = PathBuf::from("/tmp").join("rustmesh-openmesh-decimation-trace.cpp");
    let bin_path = PathBuf::from("/tmp").join("rustmesh-openmesh-decimation-trace");
    fs::write(&cpp_path, openmesh_trace_cpp_source())?;

    let openmesh_root = openmesh_root();
    let include_src = openmesh_root.join("src");
    let include_build = openmesh_root.join("build/src");
    let lib_dir = openmesh_root.join("build/Build/lib");

    let compile_output = Command::new("c++")
        .arg("-O2")
        .arg("-std=c++17")
        .arg("-I")
        .arg(&include_src)
        .arg("-I")
        .arg(&include_build)
        .arg(&cpp_path)
        .arg("-L")
        .arg(&lib_dir)
        .arg("-lOpenMeshTools")
        .arg("-lOpenMeshCore")
        .arg("-o")
        .arg(&bin_path)
        .output()?;
    if !compile_output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "failed to compile OpenMesh decimation trace driver:\n{}\n{}",
                String::from_utf8_lossy(&compile_output.stdout),
                String::from_utf8_lossy(&compile_output.stderr)
            ),
        ));
    }

    let output = Command::new(&bin_path)
        .arg(input_path)
        .arg(target_vertices.to_string())
        .arg(trace_limit.to_string())
        .env("DYLD_LIBRARY_PATH", &lib_dir)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }

    parse_openmesh_trace(&String::from_utf8_lossy(&output.stdout))
}

fn parse_openmesh_trace(stdout: &str) -> io::Result<TraceRun> {
    let mut run = TraceRun::default();
    let allow_debug_only = std::env::var("RUSTMESH_TRACE_DUMP_VERTICES")
        .ok()
        .is_some_and(|raw| !raw.trim().is_empty());

    for line in stdout.lines() {
        if line.starts_with("OPENMESH_SUPPORT ")
            || line.starts_with("OPENMESH_HEAP ")
            || line.starts_with("  pos=")
            || line.starts_with("after ")
            || line.starts_with("vertex ")
            || line.starts_with("face ")
            || line.starts_with("  face_ops")
            || line.starts_with("  point_bits=")
            || line.starts_with("  quadric=")
            || line.starts_with("  quadric_bits=")
            || line.starts_with("  face_quadric_bits=")
            || line.starts_with("  heh=")
        {
            println!("{line}");
            continue;
        }
        if let Some(rest) = line.strip_prefix("SUMMARY ") {
            for (key, value) in parse_kv_fields(rest) {
                match key {
                    "collapsed" => run.collapsed = value.parse().unwrap_or(0),
                    "boundary" => run.boundary_collapses = value.parse().unwrap_or(0),
                    "interior" => run.interior_collapses = value.parse().unwrap_or(0),
                    "final_vertices" => run.final_vertices = value.parse().unwrap_or(0),
                    "final_faces" => run.final_faces = value.parse().unwrap_or(0),
                    _ => {}
                }
            }
        } else if let Some(rest) = line.strip_prefix("TRACE ") {
            let mut step = TraceStepRow {
                step: 0,
                removed: 0,
                kept: 0,
                boundary: false,
                faces_removed: 0,
                priority: 0.0,
                active_faces_before: 0,
                active_faces_after: 0,
            };

            for (key, value) in parse_kv_fields(rest) {
                match key {
                    "step" => step.step = value.parse().unwrap_or(0),
                    "removed" => step.removed = value.parse().unwrap_or(0),
                    "kept" => step.kept = value.parse().unwrap_or(0),
                    "boundary" => step.boundary = value == "1",
                    "faces_removed" => step.faces_removed = value.parse().unwrap_or(0),
                    "priority" => step.priority = value.parse().unwrap_or(0.0),
                    "active_faces_before" => step.active_faces_before = value.parse().unwrap_or(0),
                    "active_faces_after" => step.active_faces_after = value.parse().unwrap_or(0),
                    _ => {}
                }
            }

            run.steps.push(step);
        }
    }

    if run.collapsed == 0 && run.steps.is_empty() && !allow_debug_only {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "failed to parse OpenMesh decimation trace output",
        ));
    }

    Ok(run)
}

fn parse_kv_fields(line: &str) -> Vec<(&str, &str)> {
    line.split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect()
}

fn same_signature(lhs: &TraceStepRow, rhs: &TraceStepRow) -> bool {
    lhs.removed == rhs.removed
        && lhs.kept == rhs.kept
        && lhs.boundary == rhs.boundary
        && lhs.faces_removed == rhs.faces_removed
}

fn same_undirected_edge(lhs: &TraceStepRow, rhs: &TraceStepRow) -> bool {
    let lhs_edge = (lhs.removed.min(lhs.kept), lhs.removed.max(lhs.kept));
    let rhs_edge = (rhs.removed.min(rhs.kept), rhs.removed.max(rhs.kept));
    lhs_edge == rhs_edge && lhs.faces_removed == rhs.faces_removed
}

fn first_boundary_mix_gap(lhs: &[TraceStepRow], rhs: &[TraceStepRow]) -> Option<usize> {
    let mut lhs_boundary = 0usize;
    let mut rhs_boundary = 0usize;

    for (left, right) in lhs.iter().zip(rhs.iter()) {
        if left.boundary {
            lhs_boundary += 1;
        }
        if right.boundary {
            rhs_boundary += 1;
        }

        if lhs_boundary != rhs_boundary {
            return Some(left.step);
        }
    }

    None
}

fn format_step(step: &TraceStepRow) -> String {
    format!(
        "{}->{} {} faces={} prio={:.6} active={}=>{}",
        step.removed,
        step.kept,
        if step.boundary { "B" } else { "I" },
        step.faces_removed,
        step.priority,
        step.active_faces_before,
        step.active_faces_after
    )
}

fn print_summary(label: &str, run: &TraceRun) {
    println!(
        "{} summary: collapsed={}, boundary={}, interior={}, final V={}, final F={}",
        label,
        run.collapsed,
        run.boundary_collapses,
        run.interior_collapses,
        run.final_vertices,
        run.final_faces
    );
}

fn openmesh_trace_cpp_source() -> &'static str {
    r#"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <cstring>
#include <string>
#include <vector>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/Status.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/System/config.h>
#include <OpenMesh/Tools/Decimater/BaseDecimaterT.hh>
#include <OpenMesh/Tools/Decimater/CollapseInfoT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <OpenMesh/Tools/Utils/HeapT.hh>

struct TraceTraits : public OpenMesh::DefaultTraits {
  VertexAttributes(OpenMesh::Attributes::Status);
  FaceAttributes(OpenMesh::Attributes::Status);
  EdgeAttributes(OpenMesh::Attributes::Status);
};

using Mesh = OpenMesh::TriMesh_ArrayKernelT<TraceTraits>;

template <typename MeshT>
class TraceDecimater : public OpenMesh::Decimater::BaseDecimaterT<MeshT> {
public:
  using Base = OpenMesh::Decimater::BaseDecimaterT<MeshT>;
  using CollapseInfo = typename Base::CollapseInfo;
  using VertexHandle = typename MeshT::VertexHandle;
  using HalfedgeHandle = typename MeshT::HalfedgeHandle;
  using Quadric = OpenMesh::Geometry::QuadricT<double>;

  struct TraceStep {
    size_t step = 0;
    unsigned removed = 0;
    unsigned kept = 0;
    bool boundary = false;
    unsigned faces_removed = 0;
    float priority = 0.0f;
    size_t active_faces_before = 0;
    size_t active_faces_after = 0;
  };

  struct TraceResult {
    size_t collapsed = 0;
    size_t boundary_collapses = 0;
    size_t interior_collapses = 0;
    size_t final_vertices = 0;
    size_t final_faces = 0;
    std::vector<TraceStep> steps;
  };

  class HeapInterface {
  public:
    HeapInterface(
        MeshT& mesh,
        OpenMesh::VPropHandleT<float> prio,
        OpenMesh::VPropHandleT<int> pos)
        : mesh_(mesh), prio_(prio), pos_(pos) {}

    bool less(VertexHandle lhs, VertexHandle rhs) {
      return mesh_.property(prio_, lhs) < mesh_.property(prio_, rhs);
    }

    bool greater(VertexHandle lhs, VertexHandle rhs) {
      return mesh_.property(prio_, lhs) > mesh_.property(prio_, rhs);
    }

    int get_heap_position(VertexHandle vh) {
      return mesh_.property(pos_, vh);
    }

    void set_heap_position(VertexHandle vh, int pos) {
      mesh_.property(pos_, vh) = pos;
    }

  private:
    MeshT& mesh_;
    OpenMesh::VPropHandleT<float> prio_;
    OpenMesh::VPropHandleT<int> pos_;
  };

  using DeciHeap = OpenMesh::Utils::HeapT<VertexHandle, HeapInterface>;

  explicit TraceDecimater(MeshT& mesh)
      : Base(mesh), mesh_(mesh) {
    mesh_.add_property(collapse_target_);
    mesh_.add_property(priority_);
    mesh_.add_property(heap_position_);
    mesh_.add_property(debug_quadric_);
  }

  ~TraceDecimater() override {
    mesh_.remove_property(collapse_target_);
    mesh_.remove_property(priority_);
    mesh_.remove_property(heap_position_);
    mesh_.remove_property(debug_quadric_);
  }

  using Base::add;
  using Base::initialize;
  using Base::module;

  TraceResult trace_to(size_t target_vertices, size_t trace_limit) {
    size_t n_collapses =
        target_vertices < mesh_.n_vertices() ? mesh_.n_vertices() - target_vertices : 0;
    return trace_decimate(n_collapses, trace_limit);
  }

public:
  std::vector<unsigned> debug_vertices() const {
    static const std::vector<unsigned> vertices = []() {
      std::vector<unsigned> values;
      if (const char* raw = std::getenv("RUSTMESH_TRACE_DUMP_VERTICES")) {
        const char* cursor = raw;
        while (*cursor != '\0') {
          char* end = nullptr;
          const unsigned long parsed = std::strtoul(cursor, &end, 10);
          if (end != cursor) {
            values.push_back(static_cast<unsigned>(parsed));
            cursor = end;
          }
          while (*cursor == ',' || *cursor == ' ' || *cursor == '\t') {
            ++cursor;
          }
          if (end == cursor && *cursor != '\0') {
            ++cursor;
          }
        }
      }
      return values;
    }();
    return vertices;
  }

  size_t debug_collapses_before() const {
    static const size_t collapses = []() {
      if (const char* raw = std::getenv("RUSTMESH_TRACE_DUMP_AFTER_COLLAPSES")) {
        return static_cast<size_t>(std::strtoull(raw, nullptr, 10));
      }
      return size_t(0);
    }();
    return collapses;
  }

  std::vector<int> debug_face_ops() const {
    static const std::vector<int> faces = []() {
      std::vector<int> values;
      if (const char* raw = std::getenv("RUSTMESH_TRACE_DUMP_FACE_OPS")) {
        const char* cursor = raw;
        while (*cursor != '\0') {
          char* end = nullptr;
          const long parsed = std::strtol(cursor, &end, 10);
          if (end != cursor) {
            values.push_back(static_cast<int>(parsed));
            cursor = end;
          }
          while (*cursor == ',' || *cursor == ' ' || *cursor == '\t') {
            ++cursor;
          }
          if (end == cursor && *cursor != '\0') {
            ++cursor;
          }
        }
      }
      return values;
    }();
    return faces;
  }

  bool should_debug_face_ops(int face_idx) const {
    const auto faces = debug_face_ops();
    return std::find(faces.begin(), faces.end(), face_idx) != faces.end();
  }

  bool should_debug_step(size_t step) const {
    static const std::vector<size_t> steps = []() {
      std::vector<size_t> values;
      if (const char* raw = std::getenv("RUSTMESH_TRACE_DEBUG_STEPS")) {
        const char* cursor = raw;
        while (*cursor != '\0') {
          char* end = nullptr;
          const unsigned long long parsed = std::strtoull(cursor, &end, 10);
          if (end != cursor) {
            values.push_back(static_cast<size_t>(parsed));
            cursor = end;
          }
          while (*cursor == ',' || *cursor == ' ' || *cursor == '\t') {
            ++cursor;
          }
          if (end == cursor && *cursor != '\0') {
            ++cursor;
          }
        }
      }
      return values;
    }();
    return std::find(steps.begin(), steps.end(), step) != steps.end();
  }

  size_t debug_top_k() const {
    static const size_t top_k = []() {
      if (const char* raw = std::getenv("RUSTMESH_TRACE_DEBUG_TOP")) {
        const unsigned long long parsed = std::strtoull(raw, nullptr, 10);
        if (parsed > 0) {
          return static_cast<size_t>(parsed);
        }
      }
      return size_t(8);
    }();
    return top_k;
  }

  void debug_heap_snapshot(const char* label, size_t step) const {
    struct Row {
      int pos = -1;
      unsigned vh = 0;
      float prio = 0.0f;
      HalfedgeHandle target;
    };

    std::vector<Row> rows;
    for (auto vh : mesh_.vertices()) {
      const int pos = mesh_.property(heap_position_, vh);
      if (pos < 0) {
        continue;
      }

      Row row;
      row.pos = pos;
      row.vh = vh.idx();
      row.prio = mesh_.property(priority_, vh);
      row.target = mesh_.property(collapse_target_, vh);
      rows.push_back(row);
    }

    std::sort(rows.begin(), rows.end(), [](const Row& lhs, const Row& rhs) {
      return lhs.pos < rhs.pos;
    });

    std::cout << "OPENMESH_HEAP " << label << " step=" << step << "\n";
    const size_t limit = std::min(debug_top_k(), rows.size());
    for (size_t i = 0; i < limit; ++i) {
      const Row& row = rows[i];
      if (row.target.is_valid()) {
        const auto from = mesh_.from_vertex_handle(row.target).idx();
        const auto to = mesh_.to_vertex_handle(row.target).idx();
        std::cout << "  pos=" << row.pos
                  << " vh=" << row.vh
                  << " prio=" << row.prio
                  << " target=" << from << " -> " << to << "\n";
      } else {
        std::cout << "  pos=" << row.pos
                  << " vh=" << row.vh
                  << " prio=" << row.prio
                  << " target=none\n";
      }
    }
  }

  void debug_vertices_after_collapses(size_t collapses_before) {
    const auto vertices = debug_vertices();
    if (vertices.empty()) {
      return;
    }

    if (collapses_before > 0) {
      trace_decimate(collapses_before, 0, false);
      std::cout << "after " << collapses_before << " collapses:\n";
    } else {
      initialize_debug_quadrics();
    }

    for (unsigned idx : vertices) {
      const auto vh = mesh_.vertex_handle(static_cast<int>(idx));
      if (!vh.is_valid()) {
        std::cout << "vertex " << idx << ": invalid\n";
        continue;
      }

      const auto anchor = mesh_.halfedge_handle(vh);
      const auto point = mesh_.point(vh);
      std::cout << "vertex " << idx
                << ": anchor=" << (anchor.is_valid() ? int(anchor.idx()) : -1)
                << " boundary_vertex=" << mesh_.is_boundary(vh)
                << " point=(" << point[0] << "," << point[1] << "," << point[2] << ")"
                << "\n";
      if (debug_dump_exact()) {
        const auto xb = bit_cast_u32(point[0]);
        const auto yb = bit_cast_u32(point[1]);
        const auto zb = bit_cast_u32(point[2]);
        std::cout << "  point_bits=[0x" << std::hex << xb
                  << ",0x" << yb
                  << ",0x" << zb
                  << std::dec << "]\n";
      }
      print_quadric("  quadric=", mesh_.property(debug_quadric_, vh));
      if (debug_dump_exact()) {
        print_quadric_bits("  quadric_bits=", mesh_.property(debug_quadric_, vh));
      }

      std::vector<HalfedgeHandle> outgoing;
      for (typename MeshT::VertexOHalfedgeIter voh_it(mesh_, vh); voh_it.is_valid(); ++voh_it) {
        outgoing.push_back(*voh_it);
      }

      if (outgoing.empty()) {
        std::cout << "  no outgoing halfedges\n";
        continue;
      }

      for (auto heh : outgoing) {
        CollapseInfo ci(mesh_, heh);
        const bool boundary =
            mesh_.is_boundary(heh) || mesh_.is_boundary(mesh_.opposite_halfedge_handle(heh));
        Quadric q = mesh_.property(debug_quadric_, ci.v0);
        q += mesh_.property(debug_quadric_, ci.v1);
        const double raw_error = q(mesh_.point(ci.v1));
        if (this->is_collapse_legal(ci)) {
          const float priority = this->collapse_priority(ci);
          const unsigned faces_removed =
              (ci.fl.is_valid() ? 1u : 0u) + (ci.fr.is_valid() ? 1u : 0u);
          std::cout << "  heh=" << heh.idx()
                    << " " << idx << "->" << mesh_.to_vertex_handle(heh).idx()
                    << " boundary=" << (boundary ? 1 : 0)
                    << " faces_removed=" << faces_removed
                    << " raw_error=" << std::scientific << std::setprecision(12) << raw_error
                    << " error=" << std::scientific << std::setprecision(12) << priority
                    << std::defaultfloat << "\n";
        } else {
          std::cout << "  heh=" << heh.idx()
                    << " " << idx << "->" << mesh_.to_vertex_handle(heh).idx()
                    << " boundary=" << (boundary ? 1 : 0)
                    << " raw_error=" << std::scientific << std::setprecision(12) << raw_error
                    << std::defaultfloat << " illegal\n";
        }
      }
    }

    debug_face_quadrics();
  }

  void heap_vertex(VertexHandle vh) {
    float prio = 0.0f;
    float best_prio = FLT_MAX;
    HalfedgeHandle heh;
    HalfedgeHandle collapse_target;

    typename MeshT::VertexOHalfedgeIter voh_it(mesh_, vh);
    for (; voh_it.is_valid(); ++voh_it) {
      heh = *voh_it;
      CollapseInfo ci(mesh_, heh);
      if (this->is_collapse_legal(ci)) {
        prio = this->collapse_priority(ci);
        if (prio >= 0.0f && prio < best_prio) {
          best_prio = prio;
          collapse_target = heh;
        }
      }
    }

    if (collapse_target.is_valid()) {
      mesh_.property(collapse_target_, vh) = collapse_target;
      mesh_.property(priority_, vh) = best_prio;

      if (heap_->is_stored(vh)) {
        heap_->update(vh);
      } else {
        heap_->insert(vh);
      }
    } else {
      if (heap_->is_stored(vh)) {
        heap_->remove(vh);
      }
      mesh_.property(collapse_target_, vh) = collapse_target;
      mesh_.property(priority_, vh) = -1.0f;
    }
  }

  size_t active_faces() const {
    size_t count = 0;
    for (auto fh : mesh_.faces()) {
      if (!mesh_.status(fh).deleted()) {
        ++count;
      }
    }
    return count;
  }

  TraceResult trace_decimate(size_t n_collapses, size_t trace_limit, bool garbage_collect = true) {
    TraceResult result;
    if (!this->is_initialized()) {
      return result;
    }

    initialize_debug_quadrics();

    using Support = std::vector<VertexHandle>;
    Support support;

    HeapInterface heap_interface(mesh_, priority_, heap_position_);
    heap_ = std::unique_ptr<DeciHeap>(new DeciHeap(heap_interface));
    heap_->reserve(mesh_.n_vertices());

    for (auto vh : mesh_.vertices()) {
      heap_->reset_heap_position(vh);
      if (!mesh_.status(vh).deleted()) {
        heap_vertex(vh);
      }
    }

    while (!heap_->empty() && result.collapsed < n_collapses) {
      VertexHandle vp = heap_->front();
      HalfedgeHandle v0v1 = mesh_.property(collapse_target_, vp);
      heap_->pop_front();

      CollapseInfo ci(mesh_, v0v1);
      if (!this->is_collapse_legal(ci)) {
        continue;
      }

      support.clear();
      for (typename MeshT::VertexVertexIter vv_it = mesh_.vv_iter(ci.v0); vv_it.is_valid(); ++vv_it) {
        support.push_back(*vv_it);
      }
      const size_t debug_step = result.collapsed + 1;
      const bool should_debug = should_debug_step(debug_step);
      if (should_debug) {
        std::cout << "OPENMESH_SUPPORT step=" << debug_step
                  << " pop=" << ci.v0.idx() << " -> " << ci.v1.idx()
                  << " support=[";
        for (size_t i = 0; i < support.size(); ++i) {
          if (i != 0) {
            std::cout << ",";
          }
          std::cout << support[i].idx();
        }
        std::cout << "]\n";
        debug_heap_snapshot("before_updates", debug_step);
      }

      const bool is_boundary = mesh_.is_boundary(ci.v0v1) || mesh_.is_boundary(ci.v1v0);
      const unsigned faces_removed =
          (ci.fl.is_valid() ? 1u : 0u) + (ci.fr.is_valid() ? 1u : 0u);
      const float priority = this->collapse_priority(ci);
      const size_t active_faces_before = active_faces();

      this->preprocess_collapse(ci);
      mesh_.collapse(v0v1);
      ++result.collapsed;
      this->postprocess_collapse(ci);
      mesh_.property(debug_quadric_, ci.v1) += mesh_.property(debug_quadric_, ci.v0);

      if (is_boundary) {
        ++result.boundary_collapses;
      } else {
        ++result.interior_collapses;
      }

      if (result.steps.size() < trace_limit) {
        TraceStep step;
        step.step = result.collapsed;
        step.removed = ci.v0.idx();
        step.kept = ci.v1.idx();
        step.boundary = is_boundary;
        step.faces_removed = faces_removed;
        step.priority = priority;
        step.active_faces_before = active_faces_before;
        step.active_faces_after = active_faces();
        result.steps.push_back(step);
      }

      for (auto vh : support) {
        if (!mesh_.status(vh).deleted()) {
          heap_vertex(vh);
        }
      }
      if (should_debug) {
        debug_heap_snapshot("after_updates", debug_step);
      }
    }

    heap_.reset();
    if (garbage_collect) {
      mesh_.garbage_collection();
      result.final_vertices = mesh_.n_vertices();
      result.final_faces = mesh_.n_faces();
    }
    return result;
  }

private:
  void initialize_debug_quadrics() {
    for (auto vh : mesh_.vertices()) {
      mesh_.property(debug_quadric_, vh).clear();
    }

    for (auto fh : mesh_.faces()) {
      typename MeshT::FaceVertexIter fv_it = mesh_.fv_iter(fh);
      const auto vh0 = *fv_it; ++fv_it;
      const auto vh1 = *fv_it; ++fv_it;
      const auto vh2 = *fv_it;

      OpenMesh::Vec3d v0 = OpenMesh::vector_cast<OpenMesh::Vec3d>(mesh_.point(vh0));
      OpenMesh::Vec3d v1 = OpenMesh::vector_cast<OpenMesh::Vec3d>(mesh_.point(vh1));
      OpenMesh::Vec3d v2 = OpenMesh::vector_cast<OpenMesh::Vec3d>(mesh_.point(vh2));

      OpenMesh::Vec3d n = (v1 - v0) % (v2 - v0);
      double area = n.norm();
      if (area > FLT_MIN) {
        n /= area;
        area *= 0.5;
      }

      Quadric q(n[0], n[1], n[2], -(v0 | n));
      q *= area;

      mesh_.property(debug_quadric_, vh0) += q;
      mesh_.property(debug_quadric_, vh1) += q;
      mesh_.property(debug_quadric_, vh2) += q;
    }
  }

  static void print_quadric(const char* prefix, const Quadric& q) {
    std::cout << prefix
              << "[" << std::scientific << std::setprecision(12)
              << q.a() << "," << q.b() << "," << q.c() << "," << q.d() << ","
              << q.e() << "," << q.f() << "," << q.g() << "," << q.h() << ","
              << q.i() << "," << q.j() << "]"
              << std::defaultfloat << "\n";
  }

  static bool debug_dump_exact() {
    static const bool enabled = []() {
      if (const char* raw = std::getenv("RUSTMESH_TRACE_DUMP_EXACT")) {
        return raw[0] != '\0' && std::string(raw) != "0";
      }
      return false;
    }();
    return enabled;
  }

  static uint32_t bit_cast_u32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  }

  static uint64_t bit_cast_u64(double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  }

  static void print_quadric_bits(const char* prefix, const Quadric& q) {
    std::cout << prefix
              << "[0x" << std::hex << std::setw(16) << std::setfill('0') << bit_cast_u64(q.a())
              << ",0x" << std::setw(16) << bit_cast_u64(q.b())
              << ",0x" << std::setw(16) << bit_cast_u64(q.c())
              << ",0x" << std::setw(16) << bit_cast_u64(q.d())
              << ",0x" << std::setw(16) << bit_cast_u64(q.e())
              << ",0x" << std::setw(16) << bit_cast_u64(q.f())
              << ",0x" << std::setw(16) << bit_cast_u64(q.g())
              << ",0x" << std::setw(16) << bit_cast_u64(q.h())
              << ",0x" << std::setw(16) << bit_cast_u64(q.i())
              << ",0x" << std::setw(16) << bit_cast_u64(q.j())
              << std::dec << std::setfill(' ') << "]\n";
  }

  static void print_face_ops(const char* prefix,
                             const OpenMesh::Vec3d& e1,
                             const OpenMesh::Vec3d& e2,
                             const OpenMesh::Vec3d& cross,
                             double sqrnorm,
                             double area_norm,
                             const OpenMesh::Vec3d& unit_normal,
                             double area_half,
                             double plane_dot,
                             double d) {
    std::cout << prefix
              << " e1=[0x" << std::hex << std::setw(16) << std::setfill('0') << bit_cast_u64(e1[0])
              << ",0x" << std::setw(16) << bit_cast_u64(e1[1])
              << ",0x" << std::setw(16) << bit_cast_u64(e1[2])
              << "] e2=[0x" << std::setw(16) << bit_cast_u64(e2[0])
              << ",0x" << std::setw(16) << bit_cast_u64(e2[1])
              << ",0x" << std::setw(16) << bit_cast_u64(e2[2])
              << "] cross=[0x" << std::setw(16) << bit_cast_u64(cross[0])
              << ",0x" << std::setw(16) << bit_cast_u64(cross[1])
              << ",0x" << std::setw(16) << bit_cast_u64(cross[2])
              << "] sqrnorm=0x" << std::setw(16) << bit_cast_u64(sqrnorm)
              << " area_norm=0x" << std::setw(16) << bit_cast_u64(area_norm)
              << " unit_normal=[0x" << std::setw(16) << bit_cast_u64(unit_normal[0])
              << ",0x" << std::setw(16) << bit_cast_u64(unit_normal[1])
              << ",0x" << std::setw(16) << bit_cast_u64(unit_normal[2])
              << "] area_half=0x" << std::setw(16) << bit_cast_u64(area_half)
              << " plane_dot=0x" << std::setw(16) << bit_cast_u64(plane_dot)
              << " d=0x" << std::setw(16) << bit_cast_u64(d)
              << std::dec << std::setfill(' ') << "\n";
  }

  void debug_face_quadrics() {
    if (!debug_dump_face_quadrics()) {
      return;
    }

    std::vector<int> face_ids;
    for (unsigned idx : debug_vertices()) {
      const auto vh = mesh_.vertex_handle(static_cast<int>(idx));
      if (!vh.is_valid()) {
        continue;
      }
      for (typename MeshT::VertexFaceIter vf_it(mesh_, vh); vf_it.is_valid(); ++vf_it) {
        face_ids.push_back(vf_it->idx());
      }
    }

    std::sort(face_ids.begin(), face_ids.end());
    face_ids.erase(std::unique(face_ids.begin(), face_ids.end()), face_ids.end());
    if (face_ids.empty()) {
      return;
    }

    for (int face_idx : face_ids) {
      const auto fh = mesh_.face_handle(face_idx);
      std::vector<int> verts;
      for (typename MeshT::FaceVertexIter fv_it(mesh_, fh); fv_it.is_valid(); ++fv_it) {
        verts.push_back(fv_it->idx());
      }
      std::cout << "face " << face_idx << ": verts=[";
      for (size_t i = 0; i < verts.size(); ++i) {
        if (i > 0) {
          std::cout << ",";
        }
        std::cout << verts[i];
      }
      std::cout << "]\n";

      if (verts.size() < 3) {
        std::cout << "  face_quadric_bits=none\n";
        continue;
      }

      typename MeshT::FaceVertexIter fv_it(mesh_, fh);
      const auto vh0 = *fv_it; ++fv_it;
      const auto vh1 = *fv_it; ++fv_it;
      const auto vh2 = *fv_it;

      using Vec3 = OpenMesh::Vec3d;
      const Vec3 v0 = OpenMesh::vector_cast<Vec3>(mesh_.point(vh0));
      const Vec3 v1 = OpenMesh::vector_cast<Vec3>(mesh_.point(vh1));
      const Vec3 v2 = OpenMesh::vector_cast<Vec3>(mesh_.point(vh2));
      const Vec3 e1 = v1 - v0;
      const Vec3 e2 = v2 - v0;
      const Vec3 cross = e1 % e2;
      Vec3 n = cross;
      double sqrnorm = n.sqrnorm();
      double area_norm = n.norm();
      double area_half = area_norm;
      if (area_half > FLT_MIN) {
        n /= area_half;
        area_half *= 0.5;
      }
      const double plane_dot = (v0 | n);
      const double d = -plane_dot;
      Quadric q(n[0], n[1], n[2], d);
      q *= area_half;
      print_quadric_bits("  face_quadric_bits=", q);
      if (should_debug_face_ops(face_idx)) {
        print_face_ops("  face_ops", e1, e2, cross, sqrnorm, area_norm, n, area_half, plane_dot, d);
      }
    }
  }

  static bool debug_dump_face_quadrics() {
    static const bool enabled = []() {
      if (const char* raw = std::getenv("RUSTMESH_TRACE_DUMP_FACE_QUADRICS")) {
        return raw[0] != '\0' && std::string(raw) != "0";
      }
      return false;
    }();
    return enabled;
  }

  MeshT& mesh_;
  std::unique_ptr<DeciHeap> heap_;
  OpenMesh::VPropHandleT<HalfedgeHandle> collapse_target_;
  OpenMesh::VPropHandleT<float> priority_;
  OpenMesh::VPropHandleT<int> heap_position_;
  OpenMesh::VPropHandleT<Quadric> debug_quadric_;
};

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: trace_driver <input.off> <target_vertices> <trace_limit>\n";
    return 1;
  }

  const std::string input_path = argv[1];
  const size_t target_vertices = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
  const size_t trace_limit = static_cast<size_t>(std::strtoull(argv[3], nullptr, 10));

  Mesh mesh;
  if (!OpenMesh::IO::read_mesh(mesh, input_path)) {
    std::cerr << "failed to read mesh: " << input_path << "\n";
    return 1;
  }

  TraceDecimater<Mesh> decimater(mesh);
  OpenMesh::Decimater::ModQuadricT<Mesh>::Handle mod_quadric;
  decimater.add(mod_quadric);
  decimater.module(mod_quadric).set_binary(false);

  if (!decimater.initialize()) {
    std::cerr << "failed to initialize decimater\n";
    return 1;
  }

  if (!decimater.debug_vertices().empty()) {
    decimater.debug_vertices_after_collapses(decimater.debug_collapses_before());
    return 0;
  }

  const auto result = decimater.trace_to(target_vertices, trace_limit);
  std::cout << std::fixed << std::setprecision(9);
  std::cout << "SUMMARY"
            << " collapsed=" << result.collapsed
            << " boundary=" << result.boundary_collapses
            << " interior=" << result.interior_collapses
            << " final_vertices=" << result.final_vertices
            << " final_faces=" << result.final_faces
            << "\n";

  for (const auto& step : result.steps) {
    std::cout << "TRACE"
              << " step=" << step.step
              << " removed=" << step.removed
              << " kept=" << step.kept
              << " boundary=" << (step.boundary ? 1 : 0)
              << " faces_removed=" << step.faces_removed
              << " priority=" << step.priority
              << " active_faces_before=" << step.active_faces_before
              << " active_faces_after=" << step.active_faces_after
              << "\n";
  }

  return 0;
}
"#
}
