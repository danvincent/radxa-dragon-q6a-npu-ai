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
    pub tools: Option<Vec<ToolDef>>,
    pub tool_choice: Option<Value>,
}

#[derive(Deserialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ToolFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    pub content: Option<String>,
    pub tool_calls: Option<Vec<RequestToolCall>>,
    pub tool_call_id: Option<String>,
}

fn deserialize_optional_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let v: Option<Value> = Option::deserialize(deserializer)?;
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        _ => Ok(None),
    }
}

#[derive(Deserialize)]
pub struct RequestToolCall {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: Option<ToolCallFunction>,
}

#[derive(Deserialize)]
pub struct ToolCallFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Serialize)]
pub struct ToolCallResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunctionResponse,
}

#[derive(Serialize)]
pub struct ToolCallFunctionResponse {
    pub name: String,
    pub arguments: String,
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

pub async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".into(),
            created: 1718000000,
            owned_by: "genie-rs".into(),
        }],
    })
}

fn model_name(state: &AppState, req: &ChatRequest) -> String {
    req.model.clone().unwrap_or(state.model_name.clone())
}

/// Attempt to match a user message to a tool and extract arguments.
/// Uses simple keyword/pattern matching on the user's last message.
fn route_user_message(tools: &[ToolDef], msg: &str) -> Option<(String, String)> {
    let msg_lower = msg.to_lowercase();

    for tool in tools {
        let name_lower = tool.function.name.to_lowercase();
        let desc_match = tool
            .function
            .description
            .as_ref()
            .map(|d| d.to_lowercase())
            .unwrap_or_default();

        // Check if message mentions tool name or keywords from description
        let name_mentioned = msg_lower.contains(&name_lower) || name_lower.split('_').all(|part| msg_lower.contains(part));
        let desc_keywords = desc_match
            .split_whitespace()
            .filter(|w| w.len() > 4)
            .any(|kw| msg_lower.contains(kw));

        if !name_mentioned && !desc_keywords {
            continue;
        }

        // Found a matching tool. Try to build arguments from the message.
        let params = tool.function.parameters.as_ref();
        let args = extract_args(params, msg);

        return Some((tool.function.name.clone(), args));
    }

    None
}

/// Extract JSON arguments from a user message based on the parameter schema.
fn extract_args(params: Option<&Value>, msg: &str) -> String {
    let schema = match params {
        Some(p) => p,
        None => return "{}".to_string(),
    };

    let props = match schema.get("properties") {
        Some(Value::Object(p)) => p,
        _ => return "{}".to_string(),
    };

    let mut args = serde_json::Map::new();

    for (prop_name, prop_schema) in props {
        let prop_type = prop_schema
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("string");

        match prop_type {
            "string" => {
                // Try to find a path-like argument in the message
                if prop_name == "path" || prop_name == "file_path" || prop_name == "file" {
                    // Extract path: look for /something after certain keywords
                    if let Some(path) = extract_path_from_msg(msg) {
                        args.insert(
                            prop_name.clone(),
                            Value::String(path),
                        );
                    } else if let Some(path) = extract_quoted_arg(msg) {
                        args.insert(
                            prop_name.clone(),
                            Value::String(path),
                        );
                    }
                } else if prop_name == "command" || prop_name == "code" || prop_name == "query" {
                    // Use the message itself for command/code/query types
                    args.insert(
                        prop_name.clone(),
                        Value::String(msg.to_string()),
                    );
                }
            }
            "number" | "integer" => {
                if let Some(n) = extract_number(msg) {
                    args.insert(prop_name.clone(), json!(n));
                }
            }
            _ => {}
        }
    }

    // If we couldn't extract any arguments, try to use the full message
    if args.is_empty() {
        // Find the first string property
        if let Some((first_name, _)) = props
            .iter()
            .find(|(_, v)| v.get("type").and_then(|t| t.as_str()) == Some("string"))
        {
            args.insert(first_name.clone(), Value::String(msg.to_string()));
        } else {
            return "{}".to_string();
        }
    }

    serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".to_string())
}

fn extract_path_from_msg(msg: &str) -> Option<String> {
    // Look for patterns like /path/to/file, ~/path, or ./path
    for word in msg.split_whitespace() {
        let trimmed = word.trim_matches(|c: char| c.is_ascii_punctuation() && c != '/' && c != '~');
        if trimmed.starts_with('/') || trimmed.starts_with("~/") || trimmed.starts_with("./") || trimmed.starts_with("../") {
            return Some(trimmed.to_string());
        }
    }
    None
}

fn extract_quoted_arg(msg: &str) -> Option<String> {
    // Look for quoted strings: 'text' or "text"
    if let Some(start) = msg.find('\'') {
        let rest = &msg[start + 1..];
        if let Some(end) = rest.find('\'') {
            return Some(rest[..end].to_string());
        }
    }
    if let Some(start) = msg.find('"') {
        let rest = &msg[start + 1..];
        if let Some(end) = rest.find('"') {
            return Some(rest[..end].to_string());
        }
    }
    None
}

fn extract_number(msg: &str) -> Option<f64> {
    for word in msg.split_whitespace() {
        if let Ok(n) = word.parse::<f64>() {
            return Some(n);
        }
    }
    None
}

/// Build the full prompt from messages and optional tool definitions.
fn build_prompt(state: &AppState, messages: &[Message], tools: Option<&Vec<ToolDef>>) -> String {
    let mut system = String::new();
    let mut last_user = String::new();
    let mut tool_context = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                system = msg.content.clone().unwrap_or_default();
            }
            "user" => {
                last_user = msg.content.clone().unwrap_or_default();
            }
            "assistant" => {
                if let Some(tc) = &msg.tool_calls {
                    for call in tc {
                        if let Some(func) = &call.function {
                            if let (Some(name), Some(args)) = (&func.name, &func.arguments) {
                                tool_context.push_str(&format!(
                                    "Assistant called function '{}' with arguments: {}\n",
                                    name, args
                                ));
                            }
                        }
                    }
                }
            }
            "tool" => {
                let content = msg.content.clone().unwrap_or_default();
                let id = msg.tool_call_id.clone().unwrap_or_default();
                tool_context.push_str(&format!(
                    "Function result (call {}):\n{}\n\n",
                    id, content
                ));
            }
            _ => {}
        }
    }

    // Append tool results as last_user
    if !tool_context.is_empty() {
        let base = last_user;
        last_user = format!(
            "{}\n\n<tool_context>\n{}Please respond based on the results above.</tool_context>",
            base, tool_context
        );
    }

    // Limit system prompt to avoid exceeding context window (~4096 tokens)
    const MAX_SYSTEM_CHARS: usize = 6000;
    if system.len() > MAX_SYSTEM_CHARS {
        let truncated: String = system.chars().take(MAX_SYSTEM_CHARS).collect();
        system = format!("{}... [system prompt truncated to fit context]", truncated.trim_end());
    }

    state
        .chat_template
        .replace("{system_prompt}", &system)
        .replace("{user_input}", &last_user)
}

/// Parse a tool call from model output (fallback).
fn parse_tool_call(text: &str) -> Option<(String, String)> {
    let trimmed = text.trim();

    // Try FUNCTION_CALL: prefix
    if let Some(start) = trimmed.find("FUNCTION_CALL:") {
        let after = trimmed[start + "FUNCTION_CALL:".len()..].trim();
        return parse_tool_json(after);
    }

    // Try <tool_call>...</tool_call>
    if let Some(start) = trimmed.find("<tool_call>") {
        let after = &trimmed[start + "<tool_call>".len()..];
        if let Some(end) = after.find("</tool_call>") {
            return parse_tool_json(after[..end].trim());
        }
    }

    // Try bare JSON object with name/arguments
    parse_tool_json(trimmed)
}

fn parse_tool_json(text: &str) -> Option<(String, String)> {
    let val: Value = serde_json::from_str(text).ok()?;
    let obj = val.as_object()?;
    let name = obj.get("name")?.as_str()?.to_string();
    let args = obj.get("arguments")?;
    let args_str = if args.is_string() {
        args.as_str().unwrap().to_string()
    } else {
        serde_json::to_string(args).ok()?
    };
    Some((name, args_str))
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
            Ok(()) => {
                result_tx.send("stop".into()).ok();
            }
            Err(_) => {
                result_tx.send("length".into()).ok();
            }
        };
    });

    result_rx
}

fn count_tokens(state: &AppState, text: &str) -> u32 {
    let guard = state.ctx.lock();
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

fn make_tool_call_response(name: String, args: String) -> Value {
    json!({
        "id": make_tool_call_id(),
        "type": "function",
        "function": { "name": name, "arguments": args }
    })
}

fn build_tool_call_response(model: &str, name: String, args: String) -> Response {
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
                "tool_calls": [make_tool_call_response(name, args)]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
    }))
    .into_response()
}

fn build_text_response(
    model: &str,
    content: String,
    finish_reason: String,
    prompt: &str,
    state: &AppState,
) -> Response {
    Json(json!({
        "id": format!("chatcmpl-{}", Uuid::new_v4()),
        "object": "chat.completion",
        "created": timestamp(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": finish_reason
        }],
        "usage": make_usage(prompt, &content, state)
    }))
    .into_response()
}

pub async fn chat_completions(
    State(state): State<AppState>,
    JsonReq(req): JsonReq<ChatRequest>,
) -> Response {
    let is_stream = req.stream.unwrap_or(false);
    let model = model_name(&state, &req);
    let has_tools = req.tools.is_some();

    // Server-side tool routing: check if user message matches a tool
    if !is_stream && has_tools {
        if let Some(user_msg) = req
            .messages
            .last()
            .filter(|m| m.role == "user")
            .and_then(|m| m.content.as_deref())
        {
            if !user_msg.is_empty() {
                if let Some((name, args)) =
                    route_user_message(req.tools.as_ref().unwrap(), user_msg)
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
                Err(_) => {
                    let _ = done_tx.send(0);
                    return;
                }
            };

            std::thread::spawn(move || {
                for msg in sync_rx {
                    if tx.blocking_send(msg).is_err() {
                        break;
                    }
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
                Ok::<_, Infallible>(
                    Event::default()
                        .data(
                            json!({
                                "id": id, "object": "chat.completion.chunk",
                                "created": timestamp(), "model": model,
                                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
                            })
                            .to_string(),
                        )
                )
            }
        });

        let token_stream = ReceiverStream::new(rx).map({
            let id = id.clone();
            let model = model.clone();
            move |token| {
                Ok::<_, Infallible>(
                    Event::default()
                        .data(
                            json!({
                                "id": id.clone(), "object": "chat.completion.chunk",
                                "created": timestamp(), "model": model.clone(),
                                "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": null}]
                            })
                            .to_string(),
                        )
                )
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
                Ok::<_, Infallible>(
                    Event::default()
                        .data(
                            json!({
                                "id": id, "object": "chat.completion.chunk",
                                "created": timestamp(), "model": model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]
                            })
                            .to_string(),
                        )
                )
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

        build_text_response(&model, content, finish_reason, &prompt, &state)
    }
}

pub async fn admin_stop(State(state): State<AppState>) -> Json<serde_json::Value> {
    let ctx = state.ctx.clone();
    tokio::task::spawn_blocking(move || {
        if let Ok(g) = ctx.lock() {
            let _ = g.stop();
        }
    });
    Json(json!({"status": "ok", "message": "stop signal sent"}))
}

pub async fn admin_clear(State(state): State<AppState>) -> Json<serde_json::Value> {
    let ctx = state.ctx.clone();
    tokio::task::spawn_blocking(move || {
        if let Ok(g) = ctx.lock() {
            let _ = g.reset();
        }
    });
    Json(json!({"status": "ok", "message": "dialog reset"}))
}

pub async fn textsplitter(JsonReq(body): JsonReq<serde_json::Value>) -> Json<serde_json::Value> {
    let text = body.get("text").and_then(|v| v.as_str()).unwrap_or("");
    let chunk_size = body
        .get("chunk_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(1000) as usize;
    let chunk_overlap = body
        .get("chunk_overlap")
        .and_then(|v| v.as_u64())
        .unwrap_or(200) as usize;

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
