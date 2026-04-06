use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::core::GaussianColorRepresentation;
use crate::render::tiled_renderer::Gaussian;

use super::{SceneIoError, SceneMetadata};

pub(super) fn load_scene_ply(path: &Path) -> Result<(Vec<Gaussian>, SceneMetadata), SceneIoError> {
    let file = File::open(path).map_err(|source| SceneIoError::Read {
        path: path.display().to_string(),
        source,
    })?;
    let reader = BufReader::new(file);

    let mut gaussians = Vec::new();
    let mut metadata = SceneMetadata::default();
    metadata.format_version = 0;
    let mut in_header = true;
    let mut expected_vertices: Option<usize> = None;
    let mut property_names = Vec::new();
    let mut explicit_representation: Option<String> = None;
    let mut explicit_sh_degree: Option<usize> = None;

    for line in reader.lines() {
        let line = line.map_err(|source| SceneIoError::Read {
            path: path.display().to_string(),
            source,
        })?;
        let trimmed = line.trim();

        if in_header {
            if trimmed.starts_with("comment format_version ") {
                metadata.format_version = trimmed["comment format_version ".len()..]
                    .trim()
                    .parse()
                    .unwrap_or(0);
            } else if trimmed.starts_with("comment iterations ") {
                metadata.iterations = trimmed["comment iterations ".len()..]
                    .trim()
                    .parse()
                    .unwrap_or(0);
            } else if trimmed.starts_with("comment final_loss ") {
                metadata.final_loss = trimmed["comment final_loss ".len()..]
                    .trim()
                    .parse()
                    .unwrap_or(0.0);
            } else if trimmed.starts_with("comment gaussian_count ") {
                metadata.gaussian_count = trimmed["comment gaussian_count ".len()..]
                    .trim()
                    .parse()
                    .unwrap_or(0);
            } else if trimmed.starts_with("comment color_representation ") {
                explicit_representation = Some(
                    trimmed["comment color_representation ".len()..]
                        .trim()
                        .to_string(),
                );
            } else if trimmed.starts_with("comment sh_degree ") {
                explicit_sh_degree = trimmed["comment sh_degree ".len()..]
                    .trim()
                    .parse::<usize>()
                    .ok();
            } else if trimmed.starts_with("element vertex ") {
                expected_vertices = trimmed["element vertex ".len()..]
                    .trim()
                    .parse::<usize>()
                    .ok();
            } else if trimmed.starts_with("property ") {
                if let Some(name) = trimmed.split_whitespace().last() {
                    property_names.push(name.to_string());
                }
            } else if trimmed == "end_header" {
                in_header = false;
                metadata.color_representation = resolve_color_representation(
                    &property_names,
                    explicit_representation.as_deref(),
                    explicit_sh_degree,
                )?;
                if metadata.format_version == 0
                    && matches!(
                        metadata.color_representation,
                        GaussianColorRepresentation::SphericalHarmonics { .. }
                    )
                {
                    log::warn!(
                        "loading scene {} without format_version comment; inferring SH layout from vertex properties",
                        path.display()
                    );
                }
            }
            continue;
        }

        if trimmed.is_empty() {
            continue;
        }

        let values: Vec<f32> = trimmed
            .split_whitespace()
            .map(|value| value.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| SceneIoError::Parse(err.to_string()))?;
        if values.len() < property_names.len() {
            return Err(SceneIoError::InvalidFormat {
                message: format!(
                    "expected at least {} values per vertex, got {}",
                    property_names.len(),
                    values.len()
                ),
            });
        }

        let index_map = property_index_map(&property_names);
        let base = |name: &str| -> Result<f32, SceneIoError> {
            let idx = required_property(&index_map, name)?;
            values
                .get(idx)
                .copied()
                .ok_or_else(|| SceneIoError::InvalidFormat {
                    message: format!("missing value for property {name}"),
                })
        };

        let mut gaussian = Gaussian::new(
            [base("x")?, base("y")?, base("z")?],
            [base("scale_x")?, base("scale_y")?, base("scale_z")?],
            [
                base("rot_w")?,
                base("rot_x")?,
                base("rot_y")?,
                base("rot_z")?,
            ],
            base("opacity")?,
            [base("color_r")?, base("color_g")?, base("color_b")?],
        );

        if let GaussianColorRepresentation::SphericalHarmonics { degree } =
            metadata.color_representation
        {
            let sh_dc = [base("sh0_r")?, base("sh0_g")?, base("sh0_b")?];
            let sh_rest_coeff_count = (degree + 1).saturating_mul(degree + 1).saturating_sub(1);
            let mut sh_rest = Vec::with_capacity(sh_rest_coeff_count * 3);
            for coeff_idx in 0..sh_rest_coeff_count {
                for channel in ["r", "g", "b"] {
                    let name = format!("sh_rest_{coeff_idx}_{channel}");
                    let idx = required_property(&index_map, &name)?;
                    sh_rest.push(values[idx]);
                }
            }
            gaussian = gaussian.with_color_state(
                metadata.color_representation,
                Some(sh_dc),
                Some(sh_rest),
            );
        }

        gaussians.push(gaussian);
    }

    if let Some(expected) = expected_vertices {
        if gaussians.len() != expected {
            return Err(SceneIoError::InvalidFormat {
                message: format!(
                    "vertex count mismatch: header {}, parsed {}",
                    expected,
                    gaussians.len()
                ),
            });
        }
    }
    if metadata.gaussian_count == 0 {
        metadata.gaussian_count = gaussians.len();
    }

    Ok((gaussians, metadata))
}

fn resolve_color_representation(
    property_names: &[String],
    explicit_representation: Option<&str>,
    explicit_sh_degree: Option<usize>,
) -> Result<GaussianColorRepresentation, SceneIoError> {
    let inferred = infer_color_representation(property_names, explicit_sh_degree)?;
    match explicit_representation {
        Some("rgb") => Ok(GaussianColorRepresentation::Rgb),
        Some("spherical_harmonics") => match inferred {
            GaussianColorRepresentation::SphericalHarmonics { degree } => {
                Ok(GaussianColorRepresentation::SphericalHarmonics { degree })
            }
            GaussianColorRepresentation::Rgb => Err(SceneIoError::InvalidFormat {
                message:
                    "scene metadata declares spherical_harmonics but SH properties are missing"
                        .to_string(),
            }),
        },
        Some(other) => Err(SceneIoError::InvalidFormat {
            message: format!("unsupported color_representation '{other}'"),
        }),
        None => Ok(inferred),
    }
}

fn infer_color_representation(
    property_names: &[String],
    explicit_sh_degree: Option<usize>,
) -> Result<GaussianColorRepresentation, SceneIoError> {
    let has_sh0 = ["sh0_r", "sh0_g", "sh0_b"]
        .iter()
        .all(|name| property_names.iter().any(|property| property == name));
    if !has_sh0 {
        return Ok(GaussianColorRepresentation::Rgb);
    }

    let sh_rest_triplets = property_names
        .iter()
        .filter(|name| name.starts_with("sh_rest_"))
        .count()
        / 3;
    let degree =
        explicit_sh_degree.unwrap_or_else(|| degree_from_rest_coeff_count(sh_rest_triplets));
    Ok(GaussianColorRepresentation::SphericalHarmonics { degree })
}

fn degree_from_rest_coeff_count(rest_coeff_count: usize) -> usize {
    let coeff_count = rest_coeff_count + 1;
    let degree = (coeff_count as f64).sqrt().round() as usize;
    degree.saturating_sub(1)
}

fn property_index_map(property_names: &[String]) -> HashMap<String, usize> {
    property_names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx))
        .collect()
}

fn required_property(
    property_map: &HashMap<String, usize>,
    name: &str,
) -> Result<usize, SceneIoError> {
    property_map
        .get(name)
        .copied()
        .ok_or_else(|| SceneIoError::InvalidFormat {
            message: format!("scene is missing required property {name}"),
        })
}
