use candle_core::{Device, Result, Tensor};
use tokenizers::Tokenizer;

const TOKENIZER_PATH: &str = "models/config/tokenizer.json";

pub fn load_tokenizer() -> Result<Tokenizer> {
    Tokenizer::from_file(TOKENIZER_PATH).map_err(|e| candle_core::Error::Msg(e.to_string()))
}

pub fn tokenize(tokenizer: &Tokenizer, prompt: &str, device: &Device) -> Result<Tensor> {
    let ids = encode(tokenizer, prompt)?;
    Tensor::new(ids.as_slice(), device)?.unsqueeze(0)
}

pub fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer
        .encode(text, true)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))
}

pub fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(ids, true)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const JAPANESE_INPUT: &str = "こんにちは、ご機嫌いかが？";

    #[test]
    fn japanese_input_tokenizes_without_unknown_tokens() {
        let tokenizer = load_tokenizer().unwrap();
        let ids = encode(&tokenizer, JAPANESE_INPUT).unwrap();
        let unknown_id = tokenizer.token_to_id("<|unk|>");

        assert!(!ids.is_empty());
        assert!(!ids.iter().any(|id| Some(*id) == unknown_id));
    }

    #[test]
    fn japanese_input_survives_encode_decode_round_trip() {
        let tokenizer = load_tokenizer().unwrap();
        let ids = encode(&tokenizer, JAPANESE_INPUT).unwrap();

        assert_eq!(decode(&tokenizer, &ids).unwrap(), JAPANESE_INPUT);
    }
}
