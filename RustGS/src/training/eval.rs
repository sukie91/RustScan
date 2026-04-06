#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FinalTrainingMetrics {
    pub(super) final_loss: f32,
    pub(super) final_step_loss: f32,
}

pub(super) fn summarize_training_metrics(
    loss_history: &[f32],
    frame_count: usize,
) -> FinalTrainingMetrics {
    FinalTrainingMetrics {
        final_loss: summarized_final_loss(loss_history, frame_count),
        final_step_loss: loss_history.last().copied().unwrap_or(0.0),
    }
}

fn summarized_final_loss(loss_history: &[f32], frame_count: usize) -> f32 {
    if loss_history.is_empty() {
        return 0.0;
    }

    let window = frame_count.max(1).min(loss_history.len());
    let start = loss_history.len() - window;
    loss_history[start..].iter().sum::<f32>() / window as f32
}

#[cfg(test)]
mod tests {
    use super::{summarize_training_metrics, summarized_final_loss, FinalTrainingMetrics};

    #[test]
    fn summarized_final_loss_uses_last_epoch_mean() {
        let history = [0.9f32, 0.8, 0.7, 0.6, 0.5];
        let final_loss = summarized_final_loss(&history, 2);
        assert!((final_loss - 0.55).abs() < 1e-6);
    }

    #[test]
    fn summarize_training_metrics_tracks_last_step_and_epoch_mean() {
        let history = [0.9f32, 0.8, 0.7, 0.6, 0.5];
        let metrics = summarize_training_metrics(&history, 2);
        assert_eq!(
            metrics,
            FinalTrainingMetrics {
                final_loss: 0.55,
                final_step_loss: 0.5,
            }
        );
    }
}
