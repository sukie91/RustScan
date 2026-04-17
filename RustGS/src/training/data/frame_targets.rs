pub(in crate::training) fn resize_rgb(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<f32> {
    if src_width == dst_width && src_height == dst_height {
        return src.to_vec();
    }

    let mut dst = vec![0.0f32; dst_width * dst_height * 3];
    for dy in 0..dst_height {
        let sy0 = dy * src_height / dst_height;
        let sy1 = ((dy + 1) * src_height / dst_height)
            .max(sy0 + 1)
            .min(src_height);
        for dx in 0..dst_width {
            let sx0 = dx * src_width / dst_width;
            let sx1 = ((dx + 1) * src_width / dst_width)
                .max(sx0 + 1)
                .min(src_width);
            let mut acc = [0.0f32; 3];
            let mut count = 0usize;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let src_idx = (sy * src_width + sx) * 3;
                    acc[0] += src[src_idx];
                    acc[1] += src[src_idx + 1];
                    acc[2] += src[src_idx + 2];
                    count += 1;
                }
            }
            let dst_idx = (dy * dst_width + dx) * 3;
            let inv = 1.0 / count.max(1) as f32;
            dst[dst_idx] = acc[0] * inv;
            dst[dst_idx + 1] = acc[1] * inv;
            dst[dst_idx + 2] = acc[2] * inv;
        }
    }
    dst
}

pub(crate) fn resize_rgb_u8_to_f32(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<f32> {
    if src_width == dst_width && src_height == dst_height {
        return src.iter().map(|value| *value as f32 / 255.0).collect();
    }

    let src_f32 = src
        .iter()
        .map(|value| *value as f32 / 255.0)
        .collect::<Vec<_>>();
    resize_rgb(&src_f32, src_width, src_height, dst_width, dst_height)
}

#[cfg(test)]
pub(in crate::training) fn resize_depth(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<f32> {
    if src_width == dst_width && src_height == dst_height {
        return src.to_vec();
    }

    let mut dst = vec![0.0f32; dst_width * dst_height];
    for dy in 0..dst_height {
        let sy0 = dy * src_height / dst_height;
        let sy1 = ((dy + 1) * src_height / dst_height)
            .max(sy0 + 1)
            .min(src_height);
        for dx in 0..dst_width {
            let sx0 = dx * src_width / dst_width;
            let sx1 = ((dx + 1) * src_width / dst_width)
                .max(sx0 + 1)
                .min(src_width);
            let mut acc = 0.0f32;
            let mut count = 0usize;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let depth = src[sy * src_width + sx];
                    if depth.is_finite() && depth > 0.0 {
                        acc += depth;
                        count += 1;
                    }
                }
            }
            dst[dy * dst_width + dx] = if count == 0 { 0.0 } else { acc / count as f32 };
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::{resize_depth, resize_rgb_u8_to_f32};

    #[test]
    fn resize_rgb_u8_to_f32_preserves_identity_dimensions() {
        let src = vec![0u8, 64, 255, 255, 128, 0];
        let resized = resize_rgb_u8_to_f32(&src, 2, 1, 2, 1);
        assert_eq!(resized.len(), 6);
        assert!((resized[0] - 0.0).abs() < 1e-6);
        assert!((resized[1] - (64.0 / 255.0)).abs() < 1e-6);
        assert!((resized[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn resize_depth_averages_only_valid_samples() {
        let src = vec![0.0, 1.0, 2.0, f32::NAN];
        let resized = resize_depth(&src, 2, 2, 1, 1);
        assert_eq!(resized.len(), 1);
        assert!((resized[0] - 1.5).abs() < 1e-6);
    }
}
