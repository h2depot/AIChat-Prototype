use std::{fs, sync::Mutex};

use candle_core::{DType, Device};
use candle_transformers::quantized_var_builder::VarBuilder;
use tauri::State;
use tokenizers::Tokenizer;

use crate::{generation, granite, tokenizer};

const MODEL_PATH: &str = "models/gguf/granite-4.1-3b-Q4_K_M.gguf";
const MODEL_CONFIG_PATH: &str = "models/config/config.json";
const MAX_NEW_TOKENS: usize = 256;

pub struct LlmPipeline {
    model: granite::Granite,
    tokenizer: Tokenizer,
    cache: granite::Cache,
    generation_config: generation::GenerationConfig,
    device: Device,
}

fn load_pipeline() -> Result<LlmPipeline, String> {
    // Candle's fast Q4 MMQ path produces substantially different Q/K projections
    // from its dequantized reference for this model. Keep the GGUF weights quantized
    // in memory, but route quantized CUDA matmuls through the fallback DMMV path.
    candle_core::quantized::cuda::set_force_dmmv(true);

    let device =
        Device::cuda_if_available(0).map_err(|e| format!("failed to initialize device: {e}"))?;

    let config_json = fs::read_to_string(MODEL_CONFIG_PATH)
        .map_err(|e| format!("failed to read model config: {e}"))?;
    let model_config: granite::GraniteConfig = serde_json::from_str(&config_json)
        .map_err(|e| format!("failed to parse model config: {e}"))?;
    let eos_token_id = match model_config.eos_token_id.as_ref() {
        Some(granite::GraniteEosToks::Single(id)) => *id,
        Some(granite::GraniteEosToks::Multiple(ids)) => *ids
            .first()
            .ok_or_else(|| "model config contains an empty EOS token list".to_string())?,
        None => return Err("model config does not define an EOS token".to_string()),
    };

    let config = model_config.into_config(false);
    let var_builder = VarBuilder::from_gguf(MODEL_PATH, &device)
        .map_err(|e| format!("failed to load GGUF weights: {e}"))?;
    let model = granite::Granite::load(var_builder, &config)
        .map_err(|e| format!("failed to build Granite model: {e}"))?;
    let tokenizer =
        tokenizer::load_tokenizer().map_err(|e| format!("failed to load tokenizer: {e}"))?;
    let cache = granite::Cache::new(config.use_cache, DType::F32, &config, &device)
        .map_err(|e| format!("failed to create KV cache: {e}"))?;

    Ok(LlmPipeline {
        model,
        tokenizer,
        cache,
        generation_config: generation::GenerationConfig::new(eos_token_id, MAX_NEW_TOKENS),
        device,
    })
}

#[tauri::command]
pub fn initialize(state: State<'_, Mutex<Option<LlmPipeline>>>) -> Result<(), String> {
    let pipeline = load_pipeline()?;
    let mut state = state
        .lock()
        .map_err(|_| "LLM state lock is poisoned".to_string())?;
    *state = Some(pipeline);
    Ok(())
}

#[tauri::command]
pub fn generate(
    prompt: &str,
    state: State<'_, Mutex<Option<LlmPipeline>>>,
) -> Result<String, String> {
    let mut state = state
        .lock()
        .map_err(|_| "LLM state lock is poisoned".to_string())?;
    let pipeline = state
        .as_mut()
        .ok_or_else(|| "LLM is not initialized".to_string())?;

    generation::generate(
        &pipeline.model,
        &pipeline.tokenizer,
        &mut pipeline.cache,
        prompt,
        &pipeline.generation_config,
        &pipeline.device,
    )
    .map_err(|e| format!("generation failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Tensor, D};
    use std::{fs::File, io::BufReader};

    #[test]
    #[ignore = "loads the 2 GB model and requires the configured inference device"]
    fn japanese_prompt_reaches_model_and_produces_a_decodable_next_token() {
        let mut pipeline = load_pipeline().unwrap();
        let prompt = generation::apply_chat_template("こんにちは、ご機嫌いかが？");
        let input = tokenizer::tokenize(&pipeline.tokenizer, &prompt, &pipeline.device).unwrap();
        let prompt_len = input.dim(1).unwrap();
        let logits = pipeline
            .model
            .forward(&input, 0, &mut pipeline.cache)
            .unwrap();
        let next_id = logits
            .squeeze(0)
            .unwrap()
            .argmax(D::Minus1)
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        let decoded = tokenizer::decode(&pipeline.tokenizer, &[next_id]).unwrap();

        println!("prompt_tokens={prompt_len} next_id={next_id} decoded={decoded:?}");
        assert_eq!(logits.dims2().unwrap(), (1, 100352));
        assert!(!decoded.is_empty());
    }

    #[test]
    #[ignore = "loads the 2 GB model and requires the configured inference device"]
    fn cached_incremental_logits_match_full_prefill() {
        let mut pipeline = load_pipeline().unwrap();
        let prompt = generation::apply_chat_template("こんにちは、ご機嫌いかが？");
        let ids = tokenizer::encode(&pipeline.tokenizer, &prompt).unwrap();

        pipeline.cache.clear();
        let full_input = Tensor::new(ids.as_slice(), &pipeline.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let full_logits = pipeline
            .model
            .forward(&full_input, 0, &mut pipeline.cache)
            .unwrap();

        pipeline.cache.clear();
        let mut incremental_logits = None;
        for (index, id) in ids.iter().enumerate() {
            let input = Tensor::new(&[*id], &pipeline.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            incremental_logits = Some(
                pipeline
                    .model
                    .forward(&input, index, &mut pipeline.cache)
                    .unwrap(),
            );
        }

        let incremental_logits = incremental_logits.unwrap();
        let max_abs_diff = (&full_logits - &incremental_logits)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        println!(
            "prompt_tokens={} max_abs_logit_diff={max_abs_diff}",
            ids.len()
        );

        assert!(
            max_abs_diff < 1e-3,
            "KV-cache logits diverged by {max_abs_diff}"
        );
    }

    #[test]
    fn print_gguf_granite_metadata() {
        let mut reader = BufReader::new(File::open(MODEL_PATH).unwrap());
        let content = candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();
        let mut entries = content
            .metadata
            .iter()
            .filter(|(key, _)| {
                key.starts_with("granite.") || key.as_str() == "general.architecture"
            })
            .collect::<Vec<_>>();
        entries.sort_by_key(|(key, _)| key.as_str());
        for (key, value) in entries {
            println!("{key}={value:?}");
        }
    }

    #[test]
    #[ignore = "loads the 2 GB model and requires the configured inference device"]
    fn locate_first_hidden_state_divergence() {
        let mut pipeline = load_pipeline().unwrap();
        let prompt = generation::apply_chat_template("こんにちは、ご機嫌いかが？");
        let ids = tokenizer::encode(&pipeline.tokenizer, &prompt).unwrap();

        pipeline.cache.clear();
        let full_input = Tensor::new(ids.as_slice(), &pipeline.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let full_trace = pipeline
            .model
            .forward_hidden_trace(&full_input, 0, &mut pipeline.cache)
            .unwrap();

        pipeline.cache.clear();
        let mut incremental_trace = Vec::new();
        for (index, id) in ids.iter().enumerate() {
            let input = Tensor::new(&[*id], &pipeline.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            incremental_trace = pipeline
                .model
                .forward_hidden_trace(&input, index, &mut pipeline.cache)
                .unwrap();
        }

        for (stage, (full, incremental)) in
            full_trace.iter().zip(incremental_trace.iter()).enumerate()
        {
            let max_abs_diff = (full - incremental)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let label = match stage {
                0 => "embedding".to_string(),
                41 => "final_norm".to_string(),
                _ => format!("block_{}", stage - 1),
            };
            println!("{label}: max_abs_diff={max_abs_diff}");
        }
    }

    #[test]
    #[ignore = "loads the 2 GB model and requires the configured inference device"]
    fn locate_first_block_divergence() {
        let mut pipeline = load_pipeline().unwrap();
        let prompt = generation::apply_chat_template("こんにちは、ご機嫌いかが？");
        let ids = tokenizer::encode(&pipeline.tokenizer, &prompt).unwrap();

        pipeline.cache.clear();
        let full_input = Tensor::new(ids.as_slice(), &pipeline.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let full_trace = pipeline
            .model
            .forward_first_block_trace(&full_input, 0, &mut pipeline.cache)
            .unwrap();

        pipeline.cache.clear();
        let mut incremental_trace = Vec::new();
        for (index, id) in ids.iter().enumerate() {
            let input = Tensor::new(&[*id], &pipeline.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            incremental_trace = pipeline
                .model
                .forward_first_block_trace(&input, index, &mut pipeline.cache)
                .unwrap();
        }

        for ((label, full), (_, incremental)) in full_trace.iter().zip(incremental_trace.iter()) {
            let max_abs_diff = (full - incremental)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            println!("{label}: max_abs_diff={max_abs_diff}");
        }
    }

    #[test]
    #[ignore = "loads the 2 GB model and requires the configured inference device"]
    fn quantized_linear_is_shape_invariant_for_identical_rows() {
        let pipeline = load_pipeline().unwrap();
        let token_id = tokenizer::encode(&pipeline.tokenizer, "こんにちは")
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let diffs = pipeline
            .model
            .first_block_projection_shape_diffs(token_id, 21, &pipeline.device)
            .unwrap();

        for (name, max_abs_diff) in &diffs {
            println!("{name}: max_abs_diff={max_abs_diff}");
        }
        assert!(
            diffs.iter().all(|(_, diff)| *diff < 1e-3),
            "quantized Linear changed its result with row count"
        );
    }

    #[test]
    fn dump_gguf_tensor_names() {
        let mut reader = BufReader::new(File::open(MODEL_PATH).unwrap());
        let content = candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();
        println!("=== GGUF Tensors ({} total) ===", content.tensor_infos.len());
        let mut names: Vec<_> = content.tensor_infos.keys().collect();
        names.sort();
        for name in &names {
            let info = &content.tensor_infos[*name];
            println!("{name}: shape={:?}, dtype={:?}", info.shape, info.ggml_dtype);
        }
        println!("\n=== Key tensor check ===");
        for key in &["output.weight", "token_embd.weight"] {
            match content.tensor_infos.get(*key) {
                Some(info) => println!("FOUND: {key} shape={:?} dtype={:?}", info.shape, info.ggml_dtype),
                None => println!("NOT FOUND: {key}"),
            }
        }
    }
}
