use candle_core::Device;
use candle_transformers::quantized_var_builder::VarBuilder;

const MODEL_PATH: &str = "src/models/llm/granite-4.1-3b-Q4_K_M.gguf";

async fn get_device() -> Result<Device, String> {
    let device = Device::cuda_if_available(0).map_err(|e| format!("デバイス初期化失敗: {e}"))?;
    Ok(device)
}

async fn load_model(device: Device) -> Result<(), String> {
    let _var_builder = VarBuilder::from_gguf(MODEL_PATH, &device)
        .map_err(|e| format!("Failed to create VarBuilder: {}", e))?;
    Ok(())
}

#[tauri::command]
pub async fn generate(prompt: &str) -> Result<String, String> {
    Ok(format!("Received prompt: {}", prompt))
}

#[tauri::command]
pub async fn initialize() -> Result<(), String> {
    let device = get_device().await?;
    load_model(device).await?;
    Ok(())
}
