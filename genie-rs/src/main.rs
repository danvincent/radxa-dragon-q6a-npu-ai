mod ffi;
mod config;
mod context;
mod server;
mod chat;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::registry::ModelRegistry;
use config::qnn_config;
use context::genie_context::GenieContext;

#[derive(Parser)]
#[command(name = "genie-rs")]
#[command(about = "OpenAI-compatible service for QNN Genie SDK")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Serve {
        #[arg(long, default_value = "0.0.0.0:8080")]
        host: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value = "models/registry.toml")]
        registry: String,
        #[arg(long)]
        working_dir: Option<String>,
    },
    List {
        #[arg(long, default_value = "models/registry.toml")]
        registry: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Serve { host, model, registry, working_dir } => {
            if let Some(ref dir) = working_dir {
                std::env::set_current_dir(dir)?;
                tracing::info!("Working directory: {}", dir);
            }

            let registry_content = std::fs::read_to_string(registry)?;
            let registry: ModelRegistry = toml::from_str(&registry_content)?;

            let model_name = model.clone().unwrap_or(registry.default_model.clone());
            let entry = registry.models.get(&model_name)
                .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in registry", model_name))?;

            tracing::info!("Loading model: {}", model_name);

            let config_json = qnn_config::generate_qnn_config(entry);

            let ctx = GenieContext::new(&config_json)?;
            tracing::info!("Model loaded successfully");

            server::run(host, ctx, entry.chat_template.clone(), model_name.clone()).await?;
        }
        Commands::List { registry } => {
            let registry_content = std::fs::read_to_string(registry)?;
            let registry: ModelRegistry = toml::from_str(&registry_content)?;
            println!("Available models:");
            for (name, entry) in &registry.models {
                println!("  {} (type: {}, context: {})", name, entry.model_type, entry.context_size);
            }
        }
    }

    Ok(())
}
