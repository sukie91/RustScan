use rustgs::{run_metal_training_benchmark, MetalTrainingBenchmarkSpec, TrainingProfile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut spec = MetalTrainingBenchmarkSpec::default();
    let mut json = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--width" => spec.width = parse_next(&mut args, "--width")?,
            "--height" => spec.height = parse_next(&mut args, "--height")?,
            "--frames" => spec.frame_count = parse_next(&mut args, "--frames")?,
            "--gaussians" => spec.gaussian_count = parse_next(&mut args, "--gaussians")?,
            "--warmup" => spec.warmup_steps = parse_next(&mut args, "--warmup")?,
            "--measure" => spec.measured_steps = parse_next(&mut args, "--measure")?,
            "--smoke-iters" => spec.smoke_iterations = parse_next(&mut args, "--smoke-iters")?,
            "--profile" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --profile"))?;
                spec.training_profile = value.parse::<TrainingProfile>()?;
            }
            "--json" => json = true,
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => {
                return Err(format!("unsupported argument '{other}'").into());
            }
        }
    }

    if !rustgs::metal_available() {
        eprintln!("Metal unavailable in current environment; benchmark skipped.");
        return Ok(());
    }

    let report = run_metal_training_benchmark(&spec)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "profile={} fixture={}x{} frames={} gaussians={}",
            report.spec.training_profile,
            report.spec.width,
            report.spec.height,
            report.spec.frame_count,
            report.spec.gaussian_count
        );
        println!(
            "avg step: {:.3} ms | forward: {:.3} ms | loss: {:.3} ms | backward: {:.3} ms | optimizer: {:.3} ms",
            report.average_step_ms,
            report.average_forward_ms,
            report.average_loss_ms,
            report.average_backward_ms,
            report.average_optimizer_ms
        );
        println!(
            "avg visible gaussians: {:.1} | avg active tiles: {:.1}",
            report.average_visible_gaussians, report.average_active_tiles
        );
        println!(
            "smoke training: {:.3} ms | final_loss: {:.6} | final_gaussians: {} | active_sh_degree: {:?}",
            report.smoke_training_ms,
            report.final_loss,
            report.final_gaussians,
            report.active_sh_degree
        );
    }

    Ok(())
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, Box<dyn std::error::Error>>
where
    T::Err: std::fmt::Display,
{
    let value = args
        .next()
        .ok_or_else(|| format!("missing value for {flag}"))?;
    value
        .parse::<T>()
        .map_err(|err| format!("invalid value for {flag}: {err}").into())
}

fn print_help() {
    println!("Usage: cargo run --example training_benchmark -- [options]");
    println!("  --width <n>         input width (default: 64)");
    println!("  --height <n>        input height (default: 64)");
    println!("  --frames <n>        frame count (default: 3)");
    println!("  --gaussians <n>     gaussian count (default: 128)");
    println!("  --warmup <n>        warmup steps (default: 2)");
    println!("  --measure <n>       measured steps (default: 5)");
    println!("  --smoke-iters <n>   smoke training iterations (default: 8)");
    println!("  --profile <name>    legacy-metal | litegs-mac-v1");
    println!("  --json              print JSON output");
}
