#[path = "candle/generation.rs"]
pub mod generation;
#[path = "candle/granite.rs"]
pub mod granite;
mod llm;
#[path = "candle/tokenizer.rs"]
pub mod tokenizer;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(std::sync::Mutex::new(None::<llm::LlmPipeline>))
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![llm::generate, llm::initialize])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
