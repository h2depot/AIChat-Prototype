use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::State;

use super::chat_recorder::{ChatMessage, ChatRecorder};

const LLM_SERVER_URL: &str = "http://127.0.0.1:8000";
const MAX_NEW_TOKENS: usize = 2048;

pub struct AppState {
    chat_recorder: Mutex<ChatRecorder>,
    http_client: reqwest::Client,
}

impl AppState {
    pub fn new(max_chat_tokens: usize) -> Self {
        Self {
            chat_recorder: Mutex::new(ChatRecorder::new(max_chat_tokens)),
            http_client: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct GenerateRequest {
    messages: Vec<ChatMessage>,
    max_context_tokens: usize,
    max_new_tokens: usize,
}

#[derive(Deserialize)]
struct GenerateResponse {
    generated_text: String,
    used_tokens: usize,
}

#[tauri::command]
pub async fn initialize(state: State<'_, AppState>) -> Result<(), String> {
    state
        .http_client
        .get(LLM_SERVER_URL)
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn generate(prompt: String, state: State<'_, AppState>) -> Result<String, String> {
    let request = {
        let recorder = state
            .chat_recorder
            .lock()
            .map_err(|error| error.to_string())?;
        GenerateRequest {
            messages: recorder.messages_with_user(prompt.clone())?,
            max_context_tokens: recorder.max_chat_tokens()?,
            max_new_tokens: MAX_NEW_TOKENS,
        }
    };

    let response = state
        .http_client
        .post(format!("{LLM_SERVER_URL}/generate"))
        .json(&request)
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?
        .json::<GenerateResponse>()
        .await
        .map_err(|error| error.to_string())?;

    let generated_text = response.generated_text;
    state
        .chat_recorder
        .lock()
        .map_err(|error| error.to_string())?
        .record_exchange(prompt, generated_text.clone(), response.used_tokens);

    Ok(generated_text)
}

#[tauri::command]
pub fn clear_chat(state: State<'_, AppState>) -> Result<(), String> {
    state
        .chat_recorder
        .lock()
        .map_err(|error| error.to_string())?
        .clear();
    Ok(())
}
