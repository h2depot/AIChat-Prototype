use candle_core::{DType, Device, Result, Tensor, D};
use tokenizers::Tokenizer;

use crate::{granite, tokenizer};

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub eos_token_id: u32,
    pub max_new_tokens: usize,
}

impl GenerationConfig {
    pub fn new(eos_token_id: u32, max_new_tokens: usize) -> Self {
        Self {
            eos_token_id,
            max_new_tokens,
        }
    }
}

/// Formats one user turn using Granite's role-control tokens and leaves the
/// assistant turn open for generation.
pub fn apply_chat_template(prompt: &str) -> String {
    format!(
        "<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n\
         <|start_of_role|>assistant<|end_of_role|>"
    )
}

/// Generates one response from `prompt` and stops at EOS or `max_new_tokens`.
///
/// This initial implementation intentionally uses greedy decoding. Sampling and
/// streaming can be added later without changing the model/cache lifecycle.
pub fn generate(
    model: &granite::Granite,
    tokenizer: &Tokenizer,
    cache: &mut granite::Cache,
    prompt: &str,
    config: &GenerationConfig,
    device: &Device,
) -> Result<String> {
    cache.clear();

    let formatted_prompt = apply_chat_template(prompt);
    let input = tokenizer::tokenize(tokenizer, &formatted_prompt, device)?;
    let prompt_len = input.dim(1)?;
    if prompt_len == 0 {
        candle_core::bail!("cannot generate from an empty token sequence")
    }

    // Prefill the KV cache with the complete prompt.
    let mut logits = model.forward(&input, 0, cache)?;
    let mut index_pos = prompt_len;
    let mut generated = Vec::with_capacity(config.max_new_tokens);

    for step in 0..config.max_new_tokens {
        let next_token = logits
            .to_dtype(DType::F32)?
            .squeeze(0)?
            .argmax(D::Minus1)?
            .to_scalar::<u32>()?;

        if next_token == config.eos_token_id {
            break;
        }
        generated.push(next_token);

        if step + 1 == config.max_new_tokens {
            break;
        }

        let token = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        logits = model.forward(&token, index_pos, cache)?;
        index_pos += 1;
    }

    decode_generated(tokenizer, &generated)
}

pub fn decode_generated(tokenizer: &Tokenizer, generated: &[u32]) -> Result<String> {
    tokenizer::decode(tokenizer, generated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_template_wraps_user_input_and_opens_assistant_turn() {
        assert_eq!(
            apply_chat_template("こんにちは、ご機嫌いかが？"),
            concat!(
                "<|start_of_role|>user<|end_of_role|>こんにちは、ご機嫌いかが？",
                "<|end_of_text|>\n",
                "<|start_of_role|>assistant<|end_of_role|>"
            )
        );
    }

    #[test]
    fn chat_template_control_tokens_are_encoded_as_special_tokens() {
        let tokenizer = tokenizer::load_tokenizer().unwrap();
        let ids = tokenizer::encode(&tokenizer, &apply_chat_template("こんにちは")).unwrap();

        assert_eq!(
            ids.first(),
            tokenizer.token_to_id("<|start_of_role|>").as_ref()
        );
        assert!(ids.contains(&tokenizer.token_to_id("<|end_of_text|>").unwrap()));
        assert_eq!(
            ids.last(),
            tokenizer.token_to_id("<|end_of_role|>").as_ref()
        );
    }

    #[test]
    fn greedy_calculation_selects_largest_logit() {
        let logits = Tensor::new(&[[-2f32, 0.25, 8.5, 3.0]], &Device::Cpu).unwrap();
        let next_token = logits
            .squeeze(0)
            .unwrap()
            .argmax(D::Minus1)
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();

        assert_eq!(next_token, 2);
    }

    #[test]
    fn generated_japanese_tokens_decode_cleanly() {
        let tokenizer = tokenizer::load_tokenizer().unwrap();
        let expected = "こんにちは！元気だよ。";
        let ids = tokenizer::encode(&tokenizer, expected).unwrap();

        assert_eq!(decode_generated(&tokenizer, &ids).unwrap(), expected);
    }
}
