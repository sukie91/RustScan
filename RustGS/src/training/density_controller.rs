//! LiteGS-compatible Density Controller implementation.
//!
//! This module replicates the behavior of LiteGS's DensityControllerOfficial
//! and DensityControllerTamingGS from `litegs/training/densify.py`.

/// Statistics helper for tracking per-Gaussian metrics during training.
///
/// Mirrors LiteGS's `StatisticsHelper` for accumulating mean2d_grad,
/// fragment_weight, fragment_err, and visibility counts.
#[derive(Debug, Clone)]
pub struct StatisticsHelper {
    /// Per-Gaussian visible count (how many times each Gaussian was rendered).
    pub visible_count: Vec<u32>,
    /// Accumulated mean2d_grad: (sum, count) for each Gaussian.
    pub mean2d_grad: Vec<(f32, f32)>,
    /// Accumulated fragment_weight: (sum, count) for each Gaussian.
    pub fragment_weight: Vec<(f32, f32)>,
    /// Accumulated fragment_err: (sum, square_sum, count) for each Gaussian.
    /// Used for variance computation: var = square_sum/count - (sum/count)^2
    pub fragment_err: Vec<(f32, f32, f32)>,
    /// Current opacity values (sigmoid space).
    pub opacity: Vec<f32>,
    /// Current max scale for each Gaussian.
    pub max_scale: Vec<f32>,
}

impl StatisticsHelper {
    pub fn new(gaussian_count: usize) -> Self {
        Self {
            visible_count: vec![0; gaussian_count],
            mean2d_grad: vec![(0.0, 0.0); gaussian_count],
            fragment_weight: vec![(0.0, 0.0); gaussian_count],
            fragment_err: vec![(0.0, 0.0, 0.0); gaussian_count],
            opacity: vec![0.0; gaussian_count],
            max_scale: vec![0.0; gaussian_count],
        }
    }

    pub fn len(&self) -> usize {
        self.visible_count.len()
    }

    pub fn is_empty(&self) -> bool {
        self.visible_count.is_empty()
    }

    /// Reset all statistics for a new densify/prune cycle.
    pub fn reset(&mut self, gaussian_count: usize) {
        self.visible_count = vec![0; gaussian_count];
        self.mean2d_grad = vec![(0.0, 0.0); gaussian_count];
        self.fragment_weight = vec![(0.0, 0.0); gaussian_count];
        self.fragment_err = vec![(0.0, 0.0, 0.0); gaussian_count];
        self.opacity = vec![0.0; gaussian_count];
        self.max_scale = vec![0.0; gaussian_count];
    }

    /// Resize to match new Gaussian count after topology changes.
    pub fn resize(&mut self, new_len: usize) {
        let old_len = self.len();
        if new_len > old_len {
            self.visible_count.extend(vec![0; new_len - old_len]);
            self.mean2d_grad.extend(vec![(0.0, 0.0); new_len - old_len]);
            self.fragment_weight.extend(vec![(0.0, 0.0); new_len - old_len]);
            self.fragment_err.extend(vec![(0.0, 0.0, 0.0); new_len - old_len]);
            self.opacity.extend(vec![0.0; new_len - old_len]);
            self.max_scale.extend(vec![0.0; new_len - old_len]);
        } else if new_len < old_len {
            self.visible_count.truncate(new_len);
            self.mean2d_grad.truncate(new_len);
            self.fragment_weight.truncate(new_len);
            self.fragment_err.truncate(new_len);
            self.opacity.truncate(new_len);
            self.max_scale.truncate(new_len);
        }
    }

    /// Update mean2d_grad for a Gaussian (accumulates for mean computation).
    pub fn update_mean2d_grad(&mut self, idx: usize, value: f32) {
        if let Some((sum, count)) = self.mean2d_grad.get_mut(idx) {
            *sum += value;
            *count += 1.0;
        }
    }

    /// Update fragment_weight for a Gaussian.
    pub fn update_fragment_weight(&mut self, idx: usize, value: f32) {
        if let Some((sum, count)) = self.fragment_weight.get_mut(idx) {
            *sum += value;
            *count += 1.0;
        }
    }

    /// Update fragment_err for a Gaussian (accumulates for variance computation).
    pub fn update_fragment_err(&mut self, idx: usize, value: f32) {
        if let Some((sum, sq_sum, count)) = self.fragment_err.get_mut(idx) {
            *sum += value;
            *sq_sum += value * value;
            *count += 1.0;
        }
    }

    /// Increment visible count for a Gaussian.
    pub fn increment_visible(&mut self, idx: usize) {
        if let Some(count) = self.visible_count.get_mut(idx) {
            *count = count.saturating_add(1);
        }
    }

    /// Get mean of mean2d_grad for a Gaussian.
    pub fn get_mean2d_grad(&self, idx: usize) -> f32 {
        self.mean2d_grad
            .get(idx)
            .map(|(sum, count)| if *count > 0.0 { sum / count } else { 0.0 })
            .unwrap_or(0.0)
    }

    /// Get mean of fragment_weight for a Gaussian.
    pub fn get_fragment_weight_mean(&self, idx: usize) -> f32 {
        self.fragment_weight
            .get(idx)
            .map(|(sum, count)| if *count > 0.0 { sum / count } else { 0.0 })
            .unwrap_or(0.0)
    }

    /// Get weight_sum = fragment_weight_mean * visible_count (LiteGS semantics).
    pub fn get_weight_sum(&self, idx: usize) -> f32 {
        let weight_mean = self.get_fragment_weight_mean(idx);
        let count = self.visible_count.get(idx).copied().unwrap_or(0) as f32;
        weight_mean * count
    }

    /// Get variance of fragment_err for a Gaussian.
    /// var = square_sum / count - (sum / count)^2
    pub fn get_fragment_err_var(&self, idx: usize) -> f32 {
        self.fragment_err
            .get(idx)
            .map(|(sum, sq_sum, count)| {
                if *count > 0.0 {
                    let mean = sum / count;
                    let sq_mean = sq_sum / count;
                    (sq_mean - mean * mean).max(0.0)
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0)
    }

    /// Get fragment count for a Gaussian.
    pub fn get_fragment_count(&self, idx: usize) -> f32 {
        self.fragment_err
            .get(idx)
            .map(|(_, _, count)| *count)
            .unwrap_or(0.0)
    }

    /// Get global culling mask (Gaussians that were never visible).
    pub fn get_global_culling(&self) -> Vec<bool> {
        self.visible_count.iter().map(|&c| c == 0).collect()
    }
}

impl Default for StatisticsHelper {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Configuration for the density controller.
#[derive(Debug, Clone)]
pub struct DensityControllerConfig {
    /// Gradient threshold for densification.
    pub densify_grad_threshold: f32,
    /// Opacity threshold for pruning.
    pub opacity_threshold: f32,
    /// Percentage of screen space that defines "small" vs "large" Gaussians.
    pub percent_dense: f32,
    /// Screen extent for scale threshold computation.
    pub screen_extent: f32,
    /// Maximum screen size for splitting.
    pub screen_size_threshold: f32,
    /// Initial point count for budget calculations.
    pub init_points_num: usize,
    /// Target number of primitives (TamingGS mode).
    pub target_primitives: usize,
    /// Epoch to start densification.
    pub densify_from: usize,
    /// Epoch to stop densification.
    pub densify_until: usize,
    /// Interval between densification steps.
    pub densification_interval: usize,
    /// Interval between opacity resets.
    pub opacity_reset_interval: usize,
    /// Opacity reset mode: "decay" or "reset".
    pub opacity_reset_mode: OpacityResetMode,
    /// Prune mode: "weight" or "threshold".
    pub prune_mode: PruneMode,
}

impl Default for DensityControllerConfig {
    fn default() -> Self {
        Self {
            densify_grad_threshold: 0.0002,
            opacity_threshold: 0.005,
            percent_dense: 0.01,
            screen_extent: 1.0,
            screen_size_threshold: 20.0,
            init_points_num: 0,
            target_primitives: 1_000_000,
            densify_from: 500,
            densify_until: 15000,
            densification_interval: 100,
            opacity_reset_interval: 3000,
            opacity_reset_mode: OpacityResetMode::Decay,
            prune_mode: PruneMode::Weight,
        }
    }
}

/// Opacity reset mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpacityResetMode {
    /// Multiply opacity by decay rate (default 0.5).
    Decay,
    /// Clamp opacity to max value (default 0.005).
    Reset,
}

/// Prune mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneMode {
    /// Prune based on fragment weight (TamingGS).
    Weight,
    /// Prune based on opacity threshold (Official).
    Threshold,
}

/// Result of a topology update operation.
#[derive(Debug, Clone, Default)]
pub struct TopologyUpdateResult {
    /// Number of Gaussians added via clone.
    pub clone_added: usize,
    /// Number of Gaussians added via split.
    pub split_added: usize,
    /// Number of Gaussians pruned.
    pub pruned: usize,
    /// New Gaussian count after update.
    pub new_count: usize,
}

/// Density controller implementing LiteGS parity.
///
/// This struct mirrors the behavior of `DensityControllerOfficial` and
/// `DensityControllerTamingGS` from LiteGS.
#[derive(Debug, Clone)]
pub struct DensityController {
    pub config: DensityControllerConfig,
    pub stats: StatisticsHelper,
    /// Whether to use TamingGS-style budget-aware densification.
    pub taming_gs_mode: bool,
}

impl DensityController {
    pub fn new(config: DensityControllerConfig, taming_gs_mode: bool) -> Self {
        let stats = StatisticsHelper::new(config.init_points_num.max(1));
        Self {
            config,
            stats,
            taming_gs_mode,
        }
    }

    /// Check if densification is active at the given epoch.
    pub fn is_densify_active(&self, epoch: usize) -> bool {
        epoch >= self.config.densify_from
            && epoch < self.config.densify_until
            && epoch % self.config.densification_interval == 0
    }

    /// Check if opacity reset should happen at the given epoch.
    pub fn should_reset_opacity(&self, epoch: usize) -> bool {
        epoch >= self.config.densify_from
            && epoch < self.config.densify_until
            && epoch % self.config.opacity_reset_interval == 0
    }

    /// Get prune mask based on opacity and visibility.
    ///
    /// LiteGS Official:
    /// - transparent = opacity < threshold
    /// - invisible = global_culling (never visible)
    /// - prune_mask = transparent | invisible
    ///
    /// LiteGS TamingGS (weight mode):
    /// - weight_sum = fragment_weight * fragment_count
    /// - prune when weight_sum == 0
    pub fn get_prune_mask(&self) -> Vec<bool> {
        let n = self.stats.len();
        let mut prune_mask = vec![false; n];

        for i in 0..n {
            let opacity = self.stats.opacity.get(i).copied().unwrap_or(0.0);
            let visible = self.stats.visible_count.get(i).copied().unwrap_or(0) > 0;

            match self.config.prune_mode {
                PruneMode::Threshold => {
                    // Official: prune if opacity < threshold or never visible
                    let transparent = opacity < self.config.opacity_threshold;
                    let invisible = !visible;
                    prune_mask[i] = transparent || invisible;
                }
                PruneMode::Weight => {
                    // TamingGS: prune when weight_sum == 0
                    let weight_sum = self.stats.get_weight_sum(i);
                    prune_mask[i] = weight_sum == 0.0;
                }
            }
        }

        prune_mask
    }

    /// Get clone mask based on gradient and scale.
    ///
    /// LiteGS: clone when grad >= threshold AND scale is small.
    pub fn get_clone_mask(&self) -> Vec<bool> {
        let n = self.stats.len();
        let mut clone_mask = vec![false; n];
        let scale_threshold = self.config.percent_dense * self.config.screen_extent;

        for i in 0..n {
            let grad = self.stats.get_mean2d_grad(i);
            let max_scale = self.stats.max_scale.get(i).copied().unwrap_or(0.0);
            let opacity = self.stats.opacity.get(i).copied().unwrap_or(0.0);

            // LiteGS: abnormal (high grad) AND tiny (small scale)
            let abnormal = grad >= self.config.densify_grad_threshold;
            let tiny = max_scale <= scale_threshold;
            let has_opacity = opacity > self.config.opacity_threshold;

            clone_mask[i] = abnormal && tiny && has_opacity;
        }

        clone_mask
    }

    /// Get split mask based on gradient and scale.
    ///
    /// LiteGS: split when grad >= threshold AND scale is large.
    pub fn get_split_mask(&self) -> Vec<bool> {
        let n = self.stats.len();
        let mut split_mask = vec![false; n];
        let scale_threshold = self.config.percent_dense * self.config.screen_extent;

        for i in 0..n {
            let grad = self.stats.get_mean2d_grad(i);
            let max_scale = self.stats.max_scale.get(i).copied().unwrap_or(0.0);
            let opacity = self.stats.opacity.get(i).copied().unwrap_or(0.0);

            // LiteGS: abnormal (high grad) AND large (big scale)
            let abnormal = grad >= self.config.densify_grad_threshold;
            let large = max_scale > scale_threshold;
            let has_opacity = opacity > self.config.opacity_threshold;

            split_mask[i] = abnormal && large && has_opacity;
        }

        split_mask
    }

    /// Compute densification score for TamingGS multinomial sampling.
    ///
    /// LiteGS TamingGS: score = variance * fragment_count * opacity^2
    pub fn get_densify_score(&self, idx: usize) -> f32 {
        let var = self.stats.get_fragment_err_var(idx);
        let frag_count = self.stats.get_fragment_count(idx);
        let opacity = self.stats.opacity.get(idx).copied().unwrap_or(0.0);

        let score = var * frag_count * opacity * opacity;
        if score.is_finite() && score > 0.0 {
            score
        } else {
            0.0
        }
    }

    /// Get all densification scores for multinomial sampling.
    pub fn get_all_densify_scores(&self) -> Vec<f32> {
        (0..self.stats.len()).map(|i| self.get_densify_score(i)).collect()
    }

    /// Compute densify budget for TamingGS mode.
    ///
    /// LiteGS TamingGS:
    /// cur_target = (target - init) / (densify_until - densify_from) * (epoch - densify_from) + init
    /// budget = min(max(cur_target - current_count, 1) + prune_count, current_count)
    pub fn compute_densify_budget(&self, current_count: usize, prune_count: usize, epoch: usize) -> usize {
        if !self.taming_gs_mode {
            // Official mode: no budget limit
            return current_count;
        }

        let init = self.config.init_points_num.max(1);
        let densify_from = self.config.densify_from;
        let densify_until = self.config.densify_until;
        let target = self.config.target_primitives;

        let span = densify_until.saturating_sub(densify_from).max(1);
        let progressed = epoch.saturating_sub(densify_from);

        let cur_target = init as f32
            + ((target.saturating_sub(init)) as f32 / span as f32) * progressed as f32;
        let cur_target = cur_target.round() as usize;

        let deficit = cur_target.saturating_sub(current_count).max(1);
        deficit.saturating_add(prune_count).min(current_count)
    }

    /// Multinomial sampling for densification candidates (TamingGS).
    ///
    /// Samples indices proportional to their scores without replacement,
    /// matching LiteGS's `torch.multinomial(weights, num_samples, replacement=False)`.
    pub fn sample_densify_candidates(&self, budget: usize) -> Vec<usize> {
        let scores = self.get_all_densify_scores();

        // Create (index, score) pairs for non-zero scores
        let candidates: Vec<(usize, f32)> = scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > 0.0 && s.is_finite())
            .map(|(i, &s)| (i, s))
            .collect();

        if candidates.is_empty() || budget == 0 {
            return Vec::new();
        }

        let budget = budget.min(candidates.len());

        // If budget equals or exceeds candidates, return all
        if budget >= candidates.len() {
            return candidates.into_iter().map(|(i, _)| i).collect();
        }

        // Multinomial sampling without replacement using the alias method
        // (reservoir sampling with weighted probabilities)
        //
        // We implement weighted random sampling without replacement by:
        // 1. Compute cumulative weights
        // 2. Sample proportional to weights, removing selected items
        //
        // For efficiency with large budgets, we use Efraimidis-Spirakis algorithm:
        // Assign each item a key = random^(1/weight), then take top-k by key.
        // This gives weighted sampling without replacement in O(n log k) time.

        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Compute keys for Efraimidis-Spirakis sampling
        // key = -ln(U) / weight where U ~ Uniform(0,1)
        // This is equivalent to random^(1/weight) but numerically more stable
        let mut keyed: Vec<(usize, f64)> = candidates
            .iter()
            .filter_map(|&(idx, score)| {
                if score > 0.0 {
                    let u: f64 = rng.gen_range(1e-10..1.0);
                    let key = -u.ln() / (score as f64);
                    Some((idx, key))
                } else {
                    None
                }
            })
            .collect();

        // Select items with smallest keys (equivalent to highest random^(1/weight))
        keyed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        keyed.truncate(budget);

        keyed.into_iter().map(|(i, _)| i).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_helper_mean() {
        let mut stats = StatisticsHelper::new(3);

        // Update mean2d_grad for index 1
        stats.update_mean2d_grad(1, 0.1);
        stats.update_mean2d_grad(1, 0.3);

        // Mean should be 0.2
        assert!((stats.get_mean2d_grad(1) - 0.2).abs() < 1e-6);
        assert_eq!(stats.get_mean2d_grad(0), 0.0);
    }

    #[test]
    fn test_statistics_helper_variance() {
        let mut stats = StatisticsHelper::new(3);

        // Update fragment_err for index 0: values 2.0, 4.0
        stats.update_fragment_err(0, 2.0);
        stats.update_fragment_err(0, 4.0);

        // Mean = 3.0, Var = ((2-3)^2 + (4-3)^2) / 2 = 1.0
        let var = stats.get_fragment_err_var(0);
        assert!((var - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_prune_mask_threshold_mode() {
        let config = DensityControllerConfig {
            opacity_threshold: 0.1,
            prune_mode: PruneMode::Threshold,
            ..Default::default()
        };
        let mut controller = DensityController::new(config, false);

        // Set up stats: index 0 has low opacity, index 1 is invisible, index 2 is healthy
        controller.stats.opacity = vec![0.05, 0.5, 0.8];
        controller.stats.visible_count = vec![1, 0, 5];
        controller.stats.resize(3);

        let mask = controller.get_prune_mask();

        // Index 0: low opacity -> prune
        assert!(mask[0]);
        // Index 1: invisible -> prune
        assert!(mask[1]);
        // Index 2: healthy -> keep
        assert!(!mask[2]);
    }

    #[test]
    fn test_prune_mask_weight_mode() {
        let config = DensityControllerConfig {
            prune_mode: PruneMode::Weight,
            ..Default::default()
        };
        let mut controller = DensityController::new(config, false);

        // Set up stats: index 0 has weight_sum > 0, index 1 has weight_sum = 0
        controller.stats.resize(2);
        controller.stats.visible_count = vec![5, 3];
        // fragment_weight sum for index 0
        controller.stats.fragment_weight[0] = (0.5, 1.0); // mean = 0.5, weight_sum = 0.5 * 5 = 2.5
        // fragment_weight for index 1 is 0
        controller.stats.fragment_weight[1] = (0.0, 0.0); // weight_sum = 0

        let mask = controller.get_prune_mask();

        // Index 0: weight_sum > 0 -> keep
        assert!(!mask[0]);
        // Index 1: weight_sum = 0 -> prune
        assert!(mask[1]);
    }

    #[test]
    fn test_clone_split_masks() {
        let config = DensityControllerConfig {
            densify_grad_threshold: 0.001,
            percent_dense: 0.01,
            screen_extent: 1.0,
            opacity_threshold: 0.1,
            ..Default::default()
        };
        let mut controller = DensityController::new(config, false);

        controller.stats.resize(3);

        // Index 0: high grad, small scale -> clone
        controller.stats.mean2d_grad[0] = (0.01, 1.0); // mean = 0.01 >= 0.001
        controller.stats.max_scale[0] = 0.005; // <= 0.01
        controller.stats.opacity[0] = 0.5;

        // Index 1: high grad, large scale -> split
        controller.stats.mean2d_grad[1] = (0.01, 1.0);
        controller.stats.max_scale[1] = 0.05; // > 0.01
        controller.stats.opacity[1] = 0.5;

        // Index 2: low grad -> neither
        controller.stats.mean2d_grad[2] = (0.0001, 1.0); // < 0.001
        controller.stats.max_scale[2] = 0.005;
        controller.stats.opacity[2] = 0.5;

        let clone_mask = controller.get_clone_mask();
        let split_mask = controller.get_split_mask();

        assert!(clone_mask[0]);
        assert!(!split_mask[0]);

        assert!(!clone_mask[1]);
        assert!(split_mask[1]);

        assert!(!clone_mask[2]);
        assert!(!split_mask[2]);
    }

    #[test]
    fn test_densify_score() {
        let config = DensityControllerConfig::default();
        let mut controller = DensityController::new(config, true);

        controller.stats.resize(2);

        // Set up stats for score computation
        // score = var * frag_count * opacity^2
        controller.stats.fragment_err[0] = (6.0, 10.0, 2.0); // mean=3, sq_mean=5, var=5-9=-4->0
        // Recompute: sum=6, sq_sum=10, count=2
        // mean = 3, sq_mean = 5, var = 5 - 9 = -4 -> clamped to 0
        controller.stats.opacity[0] = 0.5;

        // Let's set up correctly for var = 1
        // Values 2, 4: sum=6, sq_sum=4+16=20, count=2
        // mean=3, sq_mean=10, var=10-9=1
        controller.stats.fragment_err[0] = (6.0, 20.0, 2.0);
        controller.stats.opacity[0] = 0.5; // opacity^2 = 0.25
        // score = 1 * 2 * 0.25 = 0.5

        let score = controller.get_densify_score(0);
        assert!((score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_densify_budget() {
        let config = DensityControllerConfig {
            init_points_num: 100,
            target_primitives: 1000,
            densify_from: 0,
            densify_until: 100,
            ..Default::default()
        };
        let controller = DensityController::new(config, true);

        // At epoch 50: cur_target = 100 + (900/100)*50 = 550
        // If current = 400, deficit = 150
        // budget = min(150 + prune(10), 400) = 160
        let budget = controller.compute_densify_budget(400, 10, 50);
        assert_eq!(budget, 160);
    }

    #[test]
    fn test_sample_densify_candidates_multinomial() {
        let config = DensityControllerConfig::default();
        let mut controller = DensityController::new(config, true);

        controller.stats.resize(5);

        // Set scores with VERY different magnitudes to clearly test weighted sampling
        controller.stats.fragment_err[0] = (0.0, 4.0, 1.0); // var=4
        controller.stats.opacity[0] = 1.0; // score = 4
        controller.stats.visible_count[0] = 1; // frag_count=1

        controller.stats.fragment_err[1] = (0.0, 1.0, 1.0); // var=1
        controller.stats.opacity[1] = 1.0; // score = 1
        controller.stats.visible_count[1] = 1;

        controller.stats.fragment_err[2] = (0.0, 100.0, 1.0); // var=100 (MUCH higher)
        controller.stats.opacity[2] = 1.0; // score = 100
        controller.stats.visible_count[2] = 1;

        controller.stats.fragment_err[3] = (0.0, 0.0, 0.0); // var=0
        controller.stats.opacity[3] = 1.0; // score = 0 (should never be selected)

        // Debug: print actual scores
        let scores = controller.get_all_densify_scores();
        eprintln!("Scores: {:?}", scores);

        // Run sampling multiple times to check statistical distribution
        let mut counts = [0usize; 5];
        for _ in 0..1000 {
            let candidates = controller.sample_densify_candidates(2);
            assert_eq!(candidates.len(), 2);
            for &idx in &candidates {
                counts[idx] += 1;
            }
        }

        eprintln!("Counts: {:?}", counts);

        // Key properties to verify:
        // 1. Index 2 (highest score=100) should be selected most often
        // 2. Index 0 (score=4) should be second most selected
        // 3. Index 1 (score=1) should be third most selected
        // 4. Index 3 (score=0) and 4 (no stats) should never be selected
        assert!(counts[2] > counts[0], "Index 2 (score=100) should be selected more than index 0 (score=4): {} vs {}", counts[2], counts[0]);
        assert!(counts[0] > counts[1], "Index 0 (score=4) should be selected more than index 1 (score=1): {} vs {}", counts[0], counts[1]);
        assert_eq!(counts[3], 0, "Index 3 (score=0) should never be selected");
        assert_eq!(counts[4], 0, "Index 4 (no stats) should never be selected");
    }
}