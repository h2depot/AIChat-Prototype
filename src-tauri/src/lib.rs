mod ai_chat;

use ai_chat::generate::{clear_chat, generate, initialize, AppState};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::new(2048))
        .invoke_handler(tauri::generate_handler![initialize, generate, clear_chat])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
