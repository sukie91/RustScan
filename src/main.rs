mod commands;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "rustscan")]
#[command(about = "RustScan workspace reconstruction workflows", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, clap::Subcommand)]
enum Commands {
    /// Extract a triangle mesh from a trained RustGS scene.
    MeshFromGs(commands::MeshFromGsArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::MeshFromGs(args) => commands::run_mesh_from_gs(args),
    }
}
