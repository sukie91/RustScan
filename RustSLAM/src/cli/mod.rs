//! RustScan CLI entrypoint.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::config::SlamConfig;
use crate::io::video_decoder as video;

mod slam_pipeline;

/// RustScan command-line arguments.
#[derive(Parser, Debug)]
#[command(name = "rustscan", version, about = "RustScan CLI")]
struct CliArgs {
    /// Input video file path.
    #[arg(long, value_name = "FILE")]
    input: Option<PathBuf>,
    /// Output directory path.
    #[arg(long, value_name = "DIR")]
    output: Option<PathBuf>,
    /// Path to TOML configuration file.
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,
    /// Output format for results.
    #[arg(long, value_enum)]
    output_format: Option<OutputFormat>,
    /// Log verbosity level.
    #[arg(long, value_enum)]
    log_level: Option<LogLevel>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum OutputFormat {
    Json,
    Text,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Json
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    fn as_str(self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

/// RustScan configuration loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RustScanConfig {
    /// Input video file path.
    input: Option<PathBuf>,
    /// Output directory path.
    output: PathBuf,
    /// Output format for results.
    output_format: OutputFormat,
    /// Log verbosity.
    log_level: Option<LogLevel>,
    /// Non-interactive execution mode.
    non_interactive: bool,
    /// SLAM configuration block.
    slam: SlamConfig,
}

impl Default for RustScanConfig {
    fn default() -> Self {
        Self {
            input: None,
            output: PathBuf::from("./output"),
            output_format: OutputFormat::Json,
            log_level: None,
            non_interactive: true,
            slam: SlamConfig::default(),
        }
    }
}

struct ResolvedConfig {
    input: PathBuf,
    output: PathBuf,
    output_format: OutputFormat,
    slam: SlamConfig,
}

#[derive(Error, Debug)]
enum CliError {
    #[error("input path is required but was not provided")]
    InputNotProvided,
    #[error("input file not found: {0}")]
    InputMissing(PathBuf),
    #[error("input path is not a file: {0}")]
    InputNotFile(PathBuf),
    #[error("failed to read config file {path}: {source}")]
    ConfigRead { path: PathBuf, source: std::io::Error },
    #[error("failed to parse config file {path}: {source}")]
    ConfigParse { path: PathBuf, source: toml::de::Error },
    #[error("output path exists but is not a directory: {0}")]
    OutputNotDirectory(PathBuf),
    #[error("failed to create output directory {path}: {source}")]
    OutputCreate { path: PathBuf, source: std::io::Error },
    #[error("failed to write results to {path}: {source}")]
    OutputWrite { path: PathBuf, source: std::io::Error },
    #[error("pipeline failed: {0}")]
    Pipeline(String),
}

impl CliError {
    fn exit_code(&self) -> ExitCode {
        match self {
            CliError::InputNotProvided
            | CliError::InputMissing(_)
            | CliError::InputNotFile(_)
            | CliError::ConfigRead { .. }
            | CliError::ConfigParse { .. }
            | CliError::OutputNotDirectory(_) => ExitCode::from(1),
            CliError::OutputCreate { .. }
            | CliError::OutputWrite { .. }
            | CliError::Pipeline(_) => ExitCode::from(2),
        }
    }

    fn error_type(&self) -> &'static str {
        match self {
            CliError::InputNotProvided
            | CliError::InputMissing(_)
            | CliError::InputNotFile(_) => "InputError",
            CliError::ConfigRead { .. } | CliError::ConfigParse { .. } => "ConfigError",
            CliError::OutputNotDirectory(_)
            | CliError::OutputCreate { .. }
            | CliError::OutputWrite { .. } => "OutputError",
            CliError::Pipeline(_) => "PipelineError",
        }
    }

    fn component(&self) -> &'static str {
        match self {
            CliError::InputNotProvided
            | CliError::InputMissing(_)
            | CliError::InputNotFile(_) => "cli",
            CliError::ConfigRead { .. } | CliError::ConfigParse { .. } => "config",
            CliError::OutputNotDirectory(_)
            | CliError::OutputCreate { .. }
            | CliError::OutputWrite { .. } => "io",
            CliError::Pipeline(_) => "pipeline",
        }
    }

    fn suggestion(&self) -> &'static str {
        match self {
            CliError::InputNotProvided => "Pass --input <video-file> or set input in the TOML config.",
            CliError::InputMissing(_) => "Verify the input path and ensure the file exists.",
            CliError::InputNotFile(_) => "Provide a valid video file path (not a directory).",
            CliError::ConfigRead { .. } => "Verify the config path and file permissions.",
            CliError::ConfigParse { .. } => "Fix TOML syntax and ensure fields match the schema.",
            CliError::OutputNotDirectory(_) => "Choose an output path that is a directory.",
            CliError::OutputCreate { .. } => "Check write permissions or select a different output directory.",
            CliError::OutputWrite { .. } => "Ensure the output directory is writable and has free space.",
            CliError::Pipeline(_) => "Run with RUST_LOG=debug for diagnostics and verify dependencies (OpenCV, ffmpeg).",
        }
    }
}

#[derive(Debug, Serialize)]
struct MeshStats {
    vertex_count: usize,
    triangle_count: usize,
    isolated_triangle_percent: f32,
}

impl Default for MeshStats {
    fn default() -> Self {
        Self {
            vertex_count: 0,
            triangle_count: 0,
            isolated_triangle_percent: 0.0,
        }
    }
}

#[derive(Debug, Serialize)]
struct ErrorInfo {
    error_type: String,
    root_cause: String,
    component: String,
    suggestion: String,
}

#[derive(Debug, Serialize)]
struct Diagnostics {
    os: String,
    arch: String,
    rustscan_version: String,
    cwd: String,
    input: Option<String>,
    output: Option<String>,
    config: Option<String>,
}

#[derive(Debug, Serialize)]
struct ResultsJson {
    status: String,
    input: Option<String>,
    output: Option<String>,
    processing_time_ms: u128,
    camera_count: usize,
    mesh: MeshStats,
    error: Option<ErrorInfo>,
    diagnostics: Diagnostics,
}

struct PipelineReport {
    camera_count: usize,
    mesh: MeshStats,
}

pub fn run() -> ExitCode {
    let start = Instant::now();
    let cli = CliArgs::parse();
    let config_path = cli.config.clone();

    let config = match load_config(&cli) {
        Ok(config) => config,
        Err(err) => {
            return handle_error(&err, start, None, cli.output_format, cli.output, config_path);
        }
    };

    let log_level = resolve_log_level(&cli, &config);
    init_logger(&log_level);

    let mut merged = config;
    let overrides = apply_overrides(&mut merged, &cli);
    for override_entry in overrides {
        debug!("CLI override: {}", override_entry);
    }

    if !merged.non_interactive {
        warn!("non_interactive=false in config; forcing non-interactive mode");
        merged.non_interactive = true;
    }

    let resolved = match finalize_config(merged) {
        Ok(resolved) => resolved,
        Err(err) => {
            return handle_error(&err, start, None, cli.output_format, cli.output, config_path);
        }
    };

    let output_dir = match ensure_output_dir(&resolved.output) {
        Ok(dir) => dir,
        Err(err) => {
            return handle_error(&err, start, Some(&resolved), Some(resolved.output_format), Some(resolved.output.clone()), config_path);
        }
    };

    info!("Starting RustScan pipeline");
    info!("Input: {}", resolved.input.display());
    info!("Output: {}", output_dir.display());
    debug!(
        "Loaded SLAM config camera: {}x{}",
        resolved.slam.camera.width,
        resolved.slam.camera.height
    );

    let pipeline_report = match run_pipeline(&resolved) {
        Ok(report) => report,
        Err(err) => {
            return handle_error(&err, start, Some(&resolved), Some(resolved.output_format), Some(resolved.output.clone()), config_path);
        }
    };

    let results = ResultsJson {
        status: "success".to_string(),
        input: Some(resolved.input.display().to_string()),
        output: Some(output_dir.display().to_string()),
        processing_time_ms: start.elapsed().as_millis(),
        camera_count: pipeline_report.camera_count,
        mesh: pipeline_report.mesh,
        error: None,
        diagnostics: build_diagnostics(
            Some(&resolved.input),
            Some(&output_dir),
            config_path.as_deref(),
        ),
    };

    if let Err(err) = write_results(&results, &output_dir, resolved.output_format) {
        return handle_error(&err, start, Some(&resolved), Some(resolved.output_format), Some(resolved.output.clone()), config_path);
    }

    if resolved.output_format == OutputFormat::Text {
        print_text_summary(&results);
    }

    ExitCode::SUCCESS
}

fn load_config(cli: &CliArgs) -> Result<RustScanConfig, CliError> {
    if let Some(path) = &cli.config {
        let content = fs::read_to_string(path)
            .map_err(|source| CliError::ConfigRead {
                path: path.clone(),
                source,
            })?;
        let config: RustScanConfig = toml::from_str(&content)
            .map_err(|source| CliError::ConfigParse {
                path: path.clone(),
                source,
            })?;
        Ok(config)
    } else {
        Ok(RustScanConfig::default())
    }
}

fn resolve_log_level(cli: &CliArgs, config: &RustScanConfig) -> String {
    if let Some(level) = cli.log_level {
        return level.as_str().to_string();
    }

    if let Ok(level) = std::env::var("RUST_LOG") {
        if !level.trim().is_empty() {
            return level;
        }
    }

    if let Some(level) = config.log_level {
        return level.as_str().to_string();
    }

    "info".to_string()
}

fn init_logger(level: &str) {
    let mut builder = env_logger::Builder::new();
    builder.target(env_logger::Target::Stderr);
    builder.filter_level(log::LevelFilter::Info);
    builder.parse_filters(level);
    builder.format(|buf, record| {
        use std::io::Write;
        let module = record.module_path().unwrap_or(record.target());
        writeln!(
            buf,
            "{} [{}] {}: {}",
            buf.timestamp_millis(),
            record.level(),
            module,
            record.args()
        )
    });

    if let Err(err) = builder.try_init() {
        eprintln!("Failed to initialize logger: {}", err);
    }
}

fn apply_overrides(config: &mut RustScanConfig, cli: &CliArgs) -> Vec<String> {
    let mut overrides = Vec::new();

    if let Some(input) = cli.input.clone() {
        if config.input.as_ref() != Some(&input) {
            overrides.push(format!("input = {}", input.display()));
        }
        config.input = Some(input);
    }

    if let Some(output) = cli.output.clone() {
        if config.output != output {
            overrides.push(format!("output = {}", output.display()));
        }
        config.output = output;
    }

    if let Some(format) = cli.output_format {
        if config.output_format != format {
            overrides.push(format!("output_format = {:?}", format));
        }
        config.output_format = format;
    }

    if let Some(level) = cli.log_level {
        if config.log_level != Some(level) {
            overrides.push(format!("log_level = {:?}", level));
        }
        config.log_level = Some(level);
    }

    overrides
}

fn finalize_config(config: RustScanConfig) -> Result<ResolvedConfig, CliError> {
    let input = match config.input {
        Some(path) => path,
        None => return Err(CliError::InputNotProvided),
    };

    if !input.exists() {
        return Err(CliError::InputMissing(input));
    }

    let metadata = fs::metadata(&input).map_err(|_| CliError::InputMissing(input.clone()))?;
    if !metadata.is_file() {
        return Err(CliError::InputNotFile(input));
    }

    let output = config.output;

    Ok(ResolvedConfig {
        input,
        output,
        output_format: config.output_format,
        slam: config.slam,
    })
}

fn ensure_output_dir(output: &Path) -> Result<PathBuf, CliError> {
    if output.exists() {
        if output.is_dir() {
            return Ok(output.to_path_buf());
        }
        return Err(CliError::OutputNotDirectory(output.to_path_buf()));
    }

    fs::create_dir_all(output).map_err(|source| CliError::OutputCreate {
        path: output.to_path_buf(),
        source,
    })?;

    Ok(output.to_path_buf())
}

fn run_pipeline(config: &ResolvedConfig) -> Result<PipelineReport, CliError> {
    info!("Initializing video decoder");
    let mut decoder = video::VideoDecoder::open(
        &config.input,
        video::VideoDecoderConfig::default(),
    )
    .map_err(|err| CliError::Pipeline(format!("Video input error: {err}")))?;

    let info = decoder.info().clone();
    let decoder_mode = if info.hardware_accel { "hardware" } else { "software" };
    info!(
        "Video stream: {}x{}, codec={}, container={}, fps={:.2}, decoder={}, mode={}",
        info.width,
        info.height,
        info.codec,
        info.container,
        info.frame_rate,
        info.decoder,
        decoder_mode
    );

    let _ = decoder
        .frame(0)
        .map_err(|err| CliError::Pipeline(format!("Video decode error: {err}")))?;

    Ok(PipelineReport {
        camera_count: 1,
        mesh: MeshStats::default(),
    })
}

fn write_results(results: &ResultsJson, output_dir: &Path, format: OutputFormat) -> Result<(), CliError> {
    if format != OutputFormat::Json {
        return Ok(());
    }

    let path = output_dir.join("results.json");
    let payload = serde_json::to_string_pretty(results)
        .map_err(|err| CliError::Pipeline(format!("Failed to serialize results: {err}")))?;

    fs::write(&path, payload).map_err(|source| CliError::OutputWrite { path, source })?;
    info!("Results written to {}", output_dir.join("results.json").display());

    Ok(())
}

fn print_text_summary(results: &ResultsJson) {
    println!("RustScan Results");
    println!("Status: {}", results.status);
    if let Some(input) = &results.input {
        println!("Input: {}", input);
    }
    if let Some(output) = &results.output {
        println!("Output: {}", output);
    }
    println!("Processing time (ms): {}", results.processing_time_ms);
    println!("Camera count: {}", results.camera_count);
    println!("Mesh vertices: {}", results.mesh.vertex_count);
    println!("Mesh triangles: {}", results.mesh.triangle_count);
}

fn build_diagnostics(input: Option<&Path>, output: Option<&Path>, config: Option<&Path>) -> Diagnostics {
    Diagnostics {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        rustscan_version: env!("CARGO_PKG_VERSION").to_string(),
        cwd: std::env::current_dir()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|_| "<unavailable>".to_string()),
        input: input.map(|path| path.display().to_string()),
        output: output.map(|path| path.display().to_string()),
        config: config.map(|path| path.display().to_string()),
    }
}

fn handle_error(
    err: &CliError,
    start: Instant,
    resolved: Option<&ResolvedConfig>,
    output_format: Option<OutputFormat>,
    output_override: Option<PathBuf>,
    config_path: Option<PathBuf>,
) -> ExitCode {
    let input = resolved.map(|r| r.input.clone());
    let output = resolved
        .map(|r| r.output.clone())
        .or(output_override);

    let diagnostics = build_diagnostics(
        input.as_deref(),
        output.as_deref(),
        config_path.as_deref(),
    );

    let error_info = ErrorInfo {
        error_type: err.error_type().to_string(),
        root_cause: err.to_string(),
        component: err.component().to_string(),
        suggestion: err.suggestion().to_string(),
    };

    error!(
        "{}: {} (component: {})",
        error_info.error_type,
        error_info.root_cause,
        error_info.component
    );

    eprintln!("Error: {}", error_info.error_type);
    eprintln!("Component: {}", error_info.component);
    eprintln!("Cause: {}", error_info.root_cause);
    eprintln!("Suggestion: {}", error_info.suggestion);
    eprintln!(
        "Diagnostics: os={}, arch={}, cwd={}",
        diagnostics.os, diagnostics.arch, diagnostics.cwd
    );
    if let Some(input) = &diagnostics.input {
        eprintln!("Diagnostics: input={}", input);
    }
    if let Some(output) = &diagnostics.output {
        eprintln!("Diagnostics: output={}", output);
    }
    if let Some(config) = &diagnostics.config {
        eprintln!("Diagnostics: config={}", config);
    }

    let results = ResultsJson {
        status: "error".to_string(),
        input: input.map(|path| path.display().to_string()),
        output: output.as_ref().map(|path| path.display().to_string()),
        processing_time_ms: start.elapsed().as_millis(),
        camera_count: 0,
        mesh: MeshStats::default(),
        error: Some(error_info),
        diagnostics,
    };

    let format = output_format.or_else(|| resolved.map(|r| r.output_format));
    if let Some(OutputFormat::Json) = format {
        if let Some(output_dir) = output {
            let _ = ensure_output_dir(&output_dir)
                .and_then(|dir| write_results(&results, &dir, OutputFormat::Json));
        }
    }

    err.exit_code()
}
