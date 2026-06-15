use crate::server::AppState;
use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        Json, Response, IntoResponse,
    },
    Json as JsonReq,
};
use serde::{Deserialize, Serialize, Deserializer};
use serde_json::{json, Value};
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
    pub max_tokens: Option<u32>,
    pub tools: Option<Vec<ToolDef>>,
}

#[derive(Deserialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub type_: String,
    pub function: ToolFunction,
}

#[derive(Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    
    pub content: Option<serde_json::Value>,
    pub tool_calls: Option<Vec<ToolCallBlock>>,
    pub tool_call_id: Option<String>,
}

fn deserialize_content<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where D: serde::Deserializer<'de> {
    use serde::de;
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::String(s) => Ok(Some(s)),
        serde_json::Value::Array(arr) => {
            // Extract text from content blocks: [{"type":"text","text":"..."}, ...]
            let mut text = String::new();
            for item in arr {
                if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                    text.push_str(t);
                    text.push('\n');
                }
            }
            if text.is_empty() { Ok(None) } else { Ok(Some(text.trim_end().to_string())) }
        }
        serde_json::Value::Null => Ok(None),
        _ => Ok(None),
    }
}

#[derive(Deserialize)]
pub struct ToolCallBlock {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub function: Option<ToolCallFunction>,
}
fn msg_content(msg: &Message) -> String {
    match &msg.content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(arr)) => {
            arr.iter()
                .filter_map(|v| v.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("
")
        }
        _ => String::new(),
    }
}


#[derive(Deserialize)]
pub struct ToolCallFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
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
        data: state.model_names.iter().map(|name| ModelInfo {
            id: name.clone(),
            object: "model".into(),
            created: 1718000000,
            owned_by: "genie-rs".into(),
        }).collect(),
    })
}

fn model_name(state: &AppState, req: &ChatRequest) -> String {
    req.model.clone().unwrap_or_else(|| state.model_name.clone())
}

fn build_prompt(state: &AppState, messages: &[Message], tools: Option<&Vec<ToolDef>>) -> String {
    const MAX_CHARS: usize = 4000; // ~1000 tokens, fits in KV cache
    let mut system = String::new();
    let mut parts: Vec<(bool, String)> = Vec::new(); // (is_user, content)
    let mut tool_context = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => { system = msg_content(msg); }
            "user" => { parts.push((true, msg_content(&msg))); }
            "assistant" => {
                if let Some(tc) = &msg.tool_calls {
                    for call in tc {
                        if let Some(func) = &call.function {
                            if let (Some(name), Some(args)) = (&func.name, &func.arguments) {
                                tool_context.push_str(&format!("Assistant called '{}' with args: {}\n", name, args));
                            }
                        }
                    }
                } else if let Some(c) = &msg.content {
                    parts.push((false, msg_content(&msg)));
                }
            }
            "tool" => {
                let c = msg_content(msg);
                let id = msg.tool_call_id.clone().unwrap_or_default();
                tool_context.push_str(&format!("Result({}):\n{}\n\n", id, c));
            }
            _ => {}
        }
    }

    // Append tool context to the last user message
    if !tool_context.is_empty() {
        if let Some((true, last)) = parts.last_mut() {
            last.push_str(&format!("\n<tool_context>\n{}Based on results above, respond.</tool_context>", tool_context));
        }
    }

    // Limit system prompt
    if system.len() > 6000 { let t: String = system.chars().take(6000).collect(); system = format!("{}... [truncated]", t.trim_end()); }

    // Find last user
    let mut last_user = String::new();
    if let Some((_, c)) = parts.iter().rev().find(|(is_user, _)| *is_user) { last_user = c.clone(); }

    // Build full prompt from scratch: system + history + last_user
    // Format: <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n...
    let mut result = String::new();
    result.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", system));

    let mut budget = MAX_CHARS;
    let mut history_turns: Vec<String> = Vec::new();
    let mut found_last = false;

    for (is_user, content) in parts.iter().rev() {
        if *is_user && !found_last { found_last = true; continue; }
        let role = if *is_user { "user" } else { "assistant" };
        history_turns.push(format!("<|im_start|>{}\n{}<|im_end|>", role, content));
    }

    // Add history (oldest to newest), trimming to fit
    for turn in history_turns.iter().rev() {
        if result.len() + turn.len() + last_user.len() + 100 > budget {
            result.push_str("<|im_start|>user\n[history truncated...]<|im_end|>\n");
            break;
        }
        result.push_str(turn);
        result.push('\n');
    }

    // Truncate last user message to fit KV cache (no echo-inducing markers)
    let remaining = budget.saturating_sub(result.len() + 50);
    let truncated_user = if last_user.len() > remaining {
        last_user.chars().take(remaining).collect()
    } else {
        last_user
    };
    result.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", truncated_user));
    result
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
        if let Err(e) = ctx_guard.reset() { tracing::warn!("dialog reset failed: {}", e); }
        tracing::info!("run_query prompt_len={}", prompt_clone.len());
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

fn make_tool_call_id() -> String {
    format!("call_{}", Uuid::new_v4().to_string().replace("-", ""))
}

fn make_usage(prompt: &str, content: &str, state: &AppState) -> Value {
    let pt = count_tokens(state, prompt);
    let ct = count_tokens(state, content);
    json!({
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct
    })
}

fn build_tool_call_response(model: &str, name: String, args: String) -> Response {
    let id = make_tool_call_id();
    Json(json!({
        "id": format!("chatcmpl-{}", Uuid::new_v4()),
        "object": "chat.completion",
        "created": timestamp(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": { "name": name, "arguments": args }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })).into_response()
}

fn parse_tool_call(text: &str) -> Option<(String, String)> {
    let trimmed = text.trim();

    // Try FUNCTION_CALL: prefix
    if let Some(start) = trimmed.find("FUNCTION_CALL:") {
        let after = trimmed[start + "FUNCTION_CALL:".len()..].trim();
        return parse_tool_json(after);
    }

    // Try <tool_call> XML tags
    if let Some(start) = trimmed.find("<tool_call>") {
        let end = trimmed.find("</tool_call>").unwrap_or(trimmed.len());
        let inner = &trimmed[start + "<tool_call>".len()..end];
        return parse_tool_json(inner.trim());
    }

    // Try bare JSON starting with {
    if trimmed.starts_with('{') {
        if let Some((name, args)) = parse_tool_json(trimmed) {
            return Some((name, args));
        }
    }

    None
}

fn extract_path_after_keyword(text: &str, key: &str) -> Option<String> {
    // Try to find the last keyword match that yields a path
    let lower = text.to_lowercase();
    // First try: match "file ", "read ", or "path " with a following absolute path
    for kw in &["file ", "path ", "read "] {
        if let Some(idx) = lower.rfind(kw) {
            let after = text[idx + kw.len()..].trim();
            if !after.is_empty() {
                let path = after.split_whitespace().next()?;
                // Prefer absolute paths (starting with /)
                if path.starts_with('/') || path.starts_with('.') || path.starts_with('~') {
                    return Some(path.to_string());
                }
            }
        }
    }
    // Fallback: take the next word after the last keyword match
    for kw in &["file ", "path ", "at ", "from ", "read "] {
        if let Some(idx) = lower.find(kw) {
            let after = text[idx + kw.len()..].trim();
            if !after.is_empty() {
                let path = after.split_whitespace().next()?;
                return Some(path.to_string());
            }
        }
    }
    None
}

fn parse_tool_json(text: &str) -> Option<(String, String)> {
    if let Ok(val) = serde_json::from_str::<Value>(text) {
        let name = val.get("name").and_then(|v| v.as_str()).or_else(|| {
            val.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str())
        })?.to_string();
        let args = val.get("arguments").or_else(|| {
            val.get("function").and_then(|f| f.get("arguments"))
        }).and_then(|v| {
            if v.is_string() { v.as_str().map(|s| s.to_string()) }
            else { Some(v.to_string()) }
        }).unwrap_or_else(|| "{}".to_string());
        Some((name, args))
    } else {
        None
    }
}

fn route_user_message(tools: &[ToolDef], user_msg: &str) -> Option<(String, String)> {
    let msg_lower = user_msg.to_lowercase();
    for tool in tools {
        let name_lower = tool.function.name.to_lowercase();
        let name_mentioned = msg_lower.contains(&name_lower)
            || name_lower.split('_').all(|part| msg_lower.contains(part));
        if name_mentioned {
            let args = extract_tool_args(user_msg, &tool.function);
            return Some((tool.function.name.clone(), args));
        }
        if let Some(desc) = &tool.function.description {
            let desc_lower = desc.to_lowercase();
            let desc_keywords: Vec<&str> = desc_lower
                .split_whitespace()
                .filter(|w| w.len() > 4)
                .collect();
            if !desc_keywords.is_empty() && desc_keywords.iter().all(|kw| msg_lower.contains(kw)) {
                let args = extract_tool_args(user_msg, &tool.function);
                return Some((tool.function.name.clone(), args));
            }
        }
    }
    None
}

fn extract_tool_args(user_msg: &str, func: &ToolFunction) -> String {
    if let Some(params) = &func.parameters {
        if let Some(props) = params.get("properties") {
            if let Some(prop_names) = props.as_object().map(|m| m.keys().cloned().collect::<Vec<_>>()) {
                if prop_names.len() == 1 {
                    let key = &prop_names[0];
                    let path_value = extract_path_after_keyword(user_msg, key);
                    if let Some(val) = path_value {
                        let mut map = serde_json::Map::new();
                        map.insert(key.clone(), Value::String(val));
                        return serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_string());
                    }
                }
            }
        }
    }
    "{}".to_string()
}



pub async fn chat_completions(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<ChatRequest>,
) -> Response {
    let is_stream = req.stream.unwrap_or(false);
    let model = model_name(&state, &req);
    let has_tools = req.tools.is_some();

    // Server-side tool routing (non-streaming only)
    if !is_stream && has_tools {
        if let Some(user_msg) = req.messages.iter()
            .last()
            .filter(|m| m.role == "user")
            .map(|m| msg_content(m))
        {
            if !user_msg.is_empty() {
                if let Some((name, args)) =
                    route_user_message(req.tools.as_ref().unwrap(), user_msg.as_str())
                {
                    return build_tool_call_response(&model, name, args);
                }
            }
        }
    }

    let prompt = build_prompt(&state, &req.messages, req.tools.as_ref());

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
            if let Err(e) = ctx_guard.reset() { tracing::warn!("dialog reset failed: {}", e); }
            tracing::info!("stream_query prompt_len={}", prompt_clone.len());

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

        // Fallback: check model output for tool calls
        if has_tools {
            if let Some((name, args)) = parse_tool_call(&content) {
                return build_tool_call_response(&model, name, args);
            }
        }

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
            "usage": make_usage(&prompt, &content, &state)
        })).into_response()
    }
}

pub async fn admin_stop(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let ctx = state.ctx.clone();
    tokio::task::spawn_blocking(move || {
        if let Ok(g) = ctx.lock() { let _ = g.stop(); }
    });
    Json(json!({"status": "ok", "message": "stop signal sent"}))
}

pub async fn admin_clear(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let ctx = state.ctx.clone();
    tokio::task::spawn_blocking(move || {
        if let Ok(g) = ctx.lock() { let _ = g.reset(); }
    });
    Json(json!({"status": "ok", "message": "dialog reset"}))
}

pub async fn textsplitter(
    JsonReq(body): JsonReq<serde_json::Value>,
) -> Json<serde_json::Value> {
    Json(json!({"status": "ok", "message": "textsplitter placeholder"}))
}

fn timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}
