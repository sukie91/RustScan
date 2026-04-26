#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SplatColorRepresentation {
    #[default]
    Rgb,
    SphericalHarmonics {
        degree: usize,
    },
}

#[cfg_attr(not(test), allow(dead_code))]
impl SplatColorRepresentation {
    pub const fn sh_degree(self) -> usize {
        match self {
            Self::Rgb => 0,
            Self::SphericalHarmonics { degree } => degree,
        }
    }
}

pub const SH_C0: f32 = 0.282_094_8;

pub const fn sh_coeff_count_for_degree(degree: usize) -> usize {
    (degree + 1) * (degree + 1)
}

pub fn rgb_to_sh0_value(rgb: f32) -> f32 {
    (rgb - 0.5) / SH_C0
}

pub fn sh0_to_rgb_value(sh: f32) -> f32 {
    (sh * SH_C0) + 0.5
}

pub fn evaluate_sh_rgb(sh_coeffs: &[f32], degree: usize, viewdir: [f32; 3]) -> [f32; 3] {
    let mut color = coeff(sh_coeffs, 0).map(|value| SH_C0 * value);
    if degree == 0 {
        return add_rgb_bias(color);
    }

    let x = viewdir[0];
    let y = viewdir[1];
    let z = viewdir[2];

    add_scaled(&mut color, coeff(sh_coeffs, 1), 0.488_602_52 * -y);
    add_scaled(&mut color, coeff(sh_coeffs, 2), 0.488_602_52 * z);
    add_scaled(&mut color, coeff(sh_coeffs, 3), 0.488_602_52 * -x);
    if degree == 1 {
        return add_rgb_bias(color);
    }

    let z2 = z * z;
    let f_tmp0_b = -1.092_548_5 * z;
    let f_tmp1_a = 0.546_274_24;
    let f_c1 = x * x - y * y;
    let f_s1 = 2.0 * x * y;
    add_scaled(&mut color, coeff(sh_coeffs, 4), f_tmp1_a * f_s1);
    add_scaled(&mut color, coeff(sh_coeffs, 5), f_tmp0_b * y);
    add_scaled(
        &mut color,
        coeff(sh_coeffs, 6),
        0.946_174_7 * z2 - 0.315_391_57,
    );
    add_scaled(&mut color, coeff(sh_coeffs, 7), f_tmp0_b * x);
    add_scaled(&mut color, coeff(sh_coeffs, 8), f_tmp1_a * f_c1);
    if degree == 2 {
        return add_rgb_bias(color);
    }

    let f_tmp0_c = -2.285_229 * z2 + 0.457_045_8;
    let f_tmp1_b = 1.445_305_7 * z;
    let f_tmp2_a = -0.590_043_6;
    let f_c2 = x * f_c1 - y * f_s1;
    let f_s2 = x * f_s1 + y * f_c1;
    add_scaled(&mut color, coeff(sh_coeffs, 9), f_tmp2_a * f_s2);
    add_scaled(&mut color, coeff(sh_coeffs, 10), f_tmp1_b * f_s1);
    add_scaled(&mut color, coeff(sh_coeffs, 11), f_tmp0_c * y);
    add_scaled(
        &mut color,
        coeff(sh_coeffs, 12),
        z * (1.865_881_7 * z2 - 1.119_529),
    );
    add_scaled(&mut color, coeff(sh_coeffs, 13), f_tmp0_c * x);
    add_scaled(&mut color, coeff(sh_coeffs, 14), f_tmp1_b * f_c1);
    add_scaled(&mut color, coeff(sh_coeffs, 15), f_tmp2_a * f_c2);
    if degree == 3 {
        return add_rgb_bias(color);
    }

    let f_tmp0_d = z * (-4.683_326 * z2 + 2.007_139_7);
    let f_tmp1_c = 3.311_611_4 * z2 - 0.473_087_34;
    let f_tmp2_b = -1.770_130_8 * z;
    let f_tmp3_a = 0.625_835_7;
    let f_c3 = x * f_c2 - y * f_s2;
    let f_s3 = x * f_s2 + y * f_c2;
    let p_sh12 = z * (1.865_881_7 * z2 - 1.119_529);
    let p_sh6 = 0.946_174_7 * z2 - 0.315_391_57;
    add_scaled(&mut color, coeff(sh_coeffs, 16), f_tmp3_a * f_s3);
    add_scaled(&mut color, coeff(sh_coeffs, 17), f_tmp2_b * f_s2);
    add_scaled(&mut color, coeff(sh_coeffs, 18), f_tmp1_c * f_s1);
    add_scaled(&mut color, coeff(sh_coeffs, 19), f_tmp0_d * y);
    add_scaled(
        &mut color,
        coeff(sh_coeffs, 20),
        1.984_313_5 * z * p_sh12 - 1.006_230_6 * p_sh6,
    );
    add_scaled(&mut color, coeff(sh_coeffs, 21), f_tmp0_d * x);
    add_scaled(&mut color, coeff(sh_coeffs, 22), f_tmp1_c * f_c1);
    add_scaled(&mut color, coeff(sh_coeffs, 23), f_tmp2_b * f_c2);
    add_scaled(&mut color, coeff(sh_coeffs, 24), f_tmp3_a * f_c3);
    add_rgb_bias(color)
}

fn coeff(sh_coeffs: &[f32], index: usize) -> [f32; 3] {
    let base = index * 3;
    [
        sh_coeffs.get(base).copied().unwrap_or_default(),
        sh_coeffs.get(base + 1).copied().unwrap_or_default(),
        sh_coeffs.get(base + 2).copied().unwrap_or_default(),
    ]
}

fn add_scaled(color: &mut [f32; 3], coeff: [f32; 3], scale: f32) {
    color[0] += coeff[0] * scale;
    color[1] += coeff[1] * scale;
    color[2] += coeff[2] * scale;
}

fn add_rgb_bias(color: [f32; 3]) -> [f32; 3] {
    [color[0] + 0.5, color[1] + 0.5, color[2] + 0.5]
}

#[cfg(test)]
mod tests {
    use super::{evaluate_sh_rgb, rgb_to_sh0_value};

    #[test]
    fn evaluate_degree_zero_matches_sh0_conversion() {
        let rgb = [0.25, 0.5, 0.75];
        let coeffs: Vec<f32> = rgb.into_iter().map(rgb_to_sh0_value).collect();

        assert_eq!(evaluate_sh_rgb(&coeffs, 0, [0.0, 0.0, 1.0]), rgb);
    }
}
