use crate::server::AppState;
use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        Json, Response, IntoResponse,
    },
    Json as JsonReq,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use std::sync::mpsc;
use futures::stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<Message>,
    pub stream: Option<bool>,
}

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

pub async fn list_models(
    State(state): State<AppState>,
) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![
            ModelInfo {
                id: state.model_name.clone(),
                object: "model".into(),
                created: 1718000000,
                owned_by: "genie-rs".into(),
            },
        ],
    })
}

fn model_name(state: &AppState, req: &ChatRequest) -> String {
    req.model.clone().unwrap_or_else(|| state.model_name.clone())
}

fn build_prompt(state: &AppState, messages: &[Message]) -> String {
    let mut system = String::new();
    let mut last_user = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => system = msg.content.clone(),
            "user" => last_user = msg.content.clone(),
            _ => {}
        }
    }

    state
        .chat_template
        .replace("{system_prompt}", &system)
        .replace("{user_input}", &last_user)
}

fn run_query(state: &AppState, prompt: &str, tx: mpsc::Sender<String>) -> mpsc::Receiver<String> {
    let (result_tx, result_rx) = mpsc::channel::<String>();
    let ctx = state.ctx.clone();
    let prompt_clone = prompt.to_string();
    let tx_clone = tx.clone();

    tokio::task::spawn_blocking(move || {
        let ctx_guard = match ctx.lock() {
            Ok(g) => g,
            Err(_) => {
                let _ = result_tx.send("stop".into());
                return;
            }
        };
        let status = ctx_guard.run_query(&prompt_clone, tx_clone);
        match status {
            Ok(()) => { result_tx.send("stop".into()).ok(); }
            Err(_) => { result_tx.send("length".into()).ok(); }
        };
    });

    result_rx
}

fn count_tokens(state: &AppState, text: &str) -> u32 {
    let ctx = state.ctx.clone();
    let guard = ctx.lock();
    match guard {
        Ok(g) => g.token_length(text).unwrap_or(0),
        Err(_) => 0,
    }
}

pub async fn chat_completions(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<ChatRequest>,
) -> Response {
    let is_stream = req.stream.unwrap_or(false);
    let prompt = build_prompt(&state, &req.messages);
    let model = model_name(&state, &req);

    if is_stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(256);
        let ctx = state.ctx.clone();
        let prompt_clone = prompt.clone();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<i32>();

        tokio::task::spawn_blocking(move || {
            let (sync_tx, sync_rx) = mpsc::channel::<String>();
            let sync_tx_for_query = sync_tx.clone();
            let ctx_guard = match ctx.lock() {
                Ok(g) => g,
                Err(_) => { let _ = done_tx.send(0); return; }
            };

            std::thread::spawn(move || {
                for msg in sync_rx {
                    if tx.blocking_send(msg).is_err() { break; }
                }
            });

            let status = ctx_guard.run_query(&prompt_clone, sync_tx_for_query);
            let code = match status {
                Ok(()) => 0,
                Err(_) => 4,
            };
            let _ = done_tx.send(code);
        });

        let id = format!("chatcmpl-{}", Uuid::new_v4());

        let init_event = futures::stream::once({
            let id = id.clone();
            let model = model.clone();
            async move {
                Ok::<_, Infallible>(Event::default().data(json!({
                    "id": id, "object": "chat.completion.chunk",
                    "created": timestamp(), "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
                }).to_string()))
            }
        });

        let token_stream = ReceiverStream::new(rx).map({
            let id = id.clone();
            let model = model.clone();
            move |token| {
                Ok::<_, Infallible>(Event::default().data(json!({
                    "id": id.clone(), "object": "chat.completion.chunk",
                    "created": timestamp(), "model": model.clone(),
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": null}]
                }).to_string()))
            }
        });

        let done_event = futures::stream::once({
            let id = id.clone();
            let model = model.clone();
            async move {
                let finish = match done_rx.await {
                    Ok(4) => "length",
                    _ => "stop",
                };
                Ok::<_, Infallible>(Event::default().data(json!({
                    "id": id, "object": "chat.completion.chunk",
                    "created": timestamp(), "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]
                }).to_string()))
            }
        });

        Sse::new(init_event.chain(token_stream).chain(done_event)).into_response()
    } else {
        let (tx, rx) = mpsc::channel();
        let result_rx = run_query(&state, &prompt, tx);

        let mut content = String::new();
        for token in rx {
            content.push_str(&token);
        }
        let finish_reason = result_rx.recv().unwrap_or("stop".into());

        let prompt_tokens = count_tokens(&state, &prompt);
        let completion_tokens = count_tokens(&state, &content);

        Json(json!({
            "id": format!("chatcmpl-{}", Uuid::new_v4()),
            "object": "chat.completion",
            "created": timestamp(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason
            }],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
        })).into_response()
    }
}

pub async fn admin_stop(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let ctx = state.ctx.clone();
    tokio::task::spawn_blocking(move || {
        if let Ok(g) = ctx.lock() {
            let _ = g.stop();
        }
    });
    Json(json!({"status": "ok", "message": "stop signal sent"}))
}

pub async fn admin_clear(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let ctx = state.ctx.clone();
    tokio::task::spawn_blocking(move || {
        if let Ok(g) = ctx.lock() {
            let _ = g.reset();
        }
    });
    Json(json!({"status": "ok", "message": "dialog reset"}))
}

pub async fn textsplitter(
    JsonReq(body): JsonReq<serde_json::Value>,
) -> Json<serde_json::Value> {
    let text = body.get("text").and_then(|v| v.as_str()).unwrap_or("");
    let chunk_size = body.get("chunk_size").and_then(|v| v.as_u64()).unwrap_or(1000) as usize;
    let chunk_overlap = body.get("chunk_overlap").and_then(|v| v.as_u64()).unwrap_or(200) as usize;

    let splitter = crate::chat::text_splitter::TextSplitter::new(chunk_size, chunk_overlap);
    let chunks = splitter.split(text);

    Json(json!({"object": "list", "data": chunks}))
}

fn timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
