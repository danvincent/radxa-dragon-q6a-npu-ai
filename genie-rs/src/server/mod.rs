pub mod routes;

use anyhow::Result;
use axum::{Router, routing::{get, post}};
use std::sync::{Arc, Mutex};
use tokio::signal;
use crate::context::genie_context::GenieContext;

#[derive(Clone)]
pub struct AppState {
    pub ctx: Arc<Mutex<GenieContext>>,
    pub chat_template: String,
    pub model_name: String,
}

pub async fn run(host: &str, ctx: GenieContext, chat_template: String, model_name: String) -> Result<()> {
    let state = AppState {
        ctx: Arc::new(Mutex::new(ctx)),
        chat_template,
        model_name,
    };

    let app = Router::new()
        .route("/", get(|| async { "Genie-RS Service" }))
        .route("/v1/models", get(routes::list_models))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/textsplitter", post(routes::textsplitter))
        .route("/v1/admin/stop", post(routes::admin_stop))
        .route("/v1/admin/clear", post(routes::admin_clear))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(host).await?;
    tracing::info!("Listening on {}", host);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => tracing::info!("Received Ctrl+C, shutting down"),
        _ = terminate => tracing::info!("Received SIGTERM, shutting down"),
    }
}
