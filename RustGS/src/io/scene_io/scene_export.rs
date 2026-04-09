use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::legacy::GaussianColorRepresentation;
use crate::render::tiled_renderer::Gaussian;

use super::{SceneIoError, SceneMetadata, CURRENT_SCENE_FORMAT_VERSION};

struct SceneColorLayout {
    representation: GaussianColorRepresentation,
    sh_rest_coeff_count: usize,
}

pub(super) fn save_scene_ply(
    path: &Path,
    gaussians: &[Gaussian],
    metadata: &SceneMetadata,
) -> Result<(), SceneIoError> {
    let layout = detect_color_layout(gaussians, metadata)?;
    let file = File::create(path).map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    let mut writer = BufWriter::new(file);

    write_line(&mut writer, path, "ply")?;
    write_line(&mut writer, path, "format ascii 1.0")?;
    write_line(&mut writer, path, "comment rustgs_scene")?;
    write_line(
        &mut writer,
        path,
        &format!("comment format_version {}", CURRENT_SCENE_FORMAT_VERSION),
    )?;
    write_line(
        &mut writer,
        path,
        &format!("comment iterations {}", metadata.iterations),
    )?;
    write_line(
        &mut writer,
        path,
        &format!("comment final_loss {}", metadata.final_loss),
    )?;
    write_line(
        &mut writer,
        path,
        &format!("comment gaussian_count {}", gaussians.len()),
    )?;
    write_line(
        &mut writer,
        path,
        &format!(
            "comment color_representation {}",
            color_representation_token(layout.representation)
        ),
    )?;
    write_line(
        &mut writer,
        path,
        &format!("comment sh_degree {}", layout.representation.sh_degree()),
    )?;
    write_line(
        &mut writer,
        path,
        &format!("element vertex {}", gaussians.len()),
    )?;

    for property in [
        "x", "y", "z", "scale_x", "scale_y", "scale_z", "rot_w", "rot_x", "rot_y", "rot_z",
        "opacity", "color_r", "color_g", "color_b",
    ] {
        write_line(&mut writer, path, &format!("property float {property}"))?;
    }
    if let GaussianColorRepresentation::SphericalHarmonics { .. } = layout.representation {
        for channel in ["r", "g", "b"] {
            write_line(&mut writer, path, &format!("property float sh0_{channel}"))?;
        }
        for coeff_idx in 0..layout.sh_rest_coeff_count {
            for channel in ["r", "g", "b"] {
                write_line(
                    &mut writer,
                    path,
                    &format!("property float sh_rest_{coeff_idx}_{channel}"),
                )?;
            }
        }
    }
    write_line(&mut writer, path, "end_header")?;

    for gaussian in gaussians {
        write!(
            writer,
            "{} {} {} {} {} {} {} {} {} {} {} {} {} {}",
            gaussian.position[0],
            gaussian.position[1],
            gaussian.position[2],
            gaussian.scale[0],
            gaussian.scale[1],
            gaussian.scale[2],
            gaussian.rotation[0],
            gaussian.rotation[1],
            gaussian.rotation[2],
            gaussian.rotation[3],
            gaussian.opacity,
            gaussian.color[0],
            gaussian.color[1],
            gaussian.color[2],
        )
        .map_err(|source| SceneIoError::Write {
            path: path.display().to_string(),
            source,
        })?;

        if let GaussianColorRepresentation::SphericalHarmonics { .. } = layout.representation {
            let sh_dc = gaussian.sh_dc.ok_or_else(|| SceneIoError::InvalidFormat {
                message: "SH scene export requires sh_dc on every gaussian".to_string(),
            })?;
            let sh_rest =
                gaussian
                    .sh_rest
                    .as_deref()
                    .ok_or_else(|| SceneIoError::InvalidFormat {
                        message: "SH scene export requires sh_rest on every gaussian".to_string(),
                    })?;
            for value in sh_dc {
                write!(writer, " {}", value).map_err(|source| SceneIoError::Write {
                    path: path.display().to_string(),
                    source,
                })?;
            }
            for value in sh_rest {
                write!(writer, " {}", value).map_err(|source| SceneIoError::Write {
                    path: path.display().to_string(),
                    source,
                })?;
            }
        }
        writeln!(writer).map_err(|source| SceneIoError::Write {
            path: path.display().to_string(),
            source,
        })?;
    }

    Ok(())
}

fn detect_color_layout(
    gaussians: &[Gaussian],
    metadata: &SceneMetadata,
) -> Result<SceneColorLayout, SceneIoError> {
    let mut representation = metadata.color_representation;
    if let Some(first) = gaussians.first() {
        representation = first.color_representation;
        for gaussian in gaussians.iter().skip(1) {
            if gaussian.color_representation != representation {
                return Err(SceneIoError::InvalidFormat {
                    message: "scene export does not support mixed color representations"
                        .to_string(),
                });
            }
        }
    }

    let sh_rest_coeff_count = match representation {
        GaussianColorRepresentation::Rgb => 0,
        GaussianColorRepresentation::SphericalHarmonics { degree } => {
            let expected = (degree + 1).saturating_mul(degree + 1).saturating_sub(1) * 3;
            for gaussian in gaussians {
                let sh_dc = gaussian.sh_dc.ok_or_else(|| SceneIoError::InvalidFormat {
                    message: "scene export requires sh_dc for SH gaussians".to_string(),
                })?;
                let sh_rest =
                    gaussian
                        .sh_rest
                        .as_deref()
                        .ok_or_else(|| SceneIoError::InvalidFormat {
                            message: "scene export requires sh_rest for SH gaussians".to_string(),
                        })?;
                let _ = sh_dc;
                if sh_rest.len() != expected {
                    return Err(SceneIoError::InvalidFormat {
                        message: format!(
                            "scene export expected {} SH-rest values for degree {}, got {}",
                            expected,
                            degree,
                            sh_rest.len()
                        ),
                    });
                }
            }
            (degree + 1).saturating_mul(degree + 1).saturating_sub(1)
        }
    };

    Ok(SceneColorLayout {
        representation,
        sh_rest_coeff_count,
    })
}

fn color_representation_token(representation: GaussianColorRepresentation) -> &'static str {
    match representation {
        GaussianColorRepresentation::Rgb => "rgb",
        GaussianColorRepresentation::SphericalHarmonics { .. } => "spherical_harmonics",
    }
}

fn write_line(writer: &mut BufWriter<File>, path: &Path, line: &str) -> Result<(), SceneIoError> {
    writeln!(writer, "{line}").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })
}
