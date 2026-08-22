//! Granite is a Long Context Transformer Language Model.
//!
//! A high performance transformer model optimized for efficient processing
//! of very long context sequences
//!
//! Excuse Me!!! This original code is from "https://github.com/huggingface/candle.git"
//! I have modified it to fit my needs, but I want to give credit to the original authors for their work and contributions to the open-source community.
//! Thank you very much!! by Helloween Head's Depot!

use candle_core::{bail, DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_transformers::{
    quantized_nn::{linear_no_bias as linear, Embedding, Linear, RmsNorm},
    quantized_var_builder::VarBuilder,
};
use std::{collections::HashMap, f32::consts::PI};

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub enum GraniteRopeType {
    #[serde(rename = "granite")]
    Granite,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct GraniteRopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: GraniteRopeType,
}
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum GraniteEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct GraniteConfig {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub attention_multiplier: f32,
    pub bos_token_id: Option<u32>,
    pub embedding_multiplier: f32,
    pub eos_token_id: Option<GraniteEosToks>,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub logits_scaling: f32,
    pub max_position_embeddings: usize,
    pub mlp_bias: bool,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: Option<usize>,
    pub pad_token_id: Option<u32>,
    pub residual_multiplier: f32,
    pub rms_norm_eps: f64,
    pub rope_scaling: Option<GraniteRopeConfig>,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: usize,
}

impl GraniteConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

fn default_rope() -> f32 {
    10_000.0
}

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
    max_position_embeddings: usize,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}
impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let theta = match &config.rope_scaling {
            None
            | Some(GraniteRopeConfig {
                rope_type: GraniteRopeType::Default,
                ..
            }) => calculate_default_inv_freq(config),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                calculate_default_inv_freq(config)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / rope_scaling.factor
                        } else {
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelen
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

        let theta = Tensor::new(theta, device)?;

        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            device: device.clone(),
            cos,
            sin,
            max_position_embeddings: config.max_position_embeddings,
        })
    }

    /// Clears all sequence-specific state so the cache can be reused.
    pub fn clear(&mut self) {
        self.kvs.iter_mut().for_each(|kv| *kv = None);
        self.masks.clear();
    }

    /// Returns the number of tokens stored in every populated layer.
    pub fn len(&self) -> Result<usize> {
        let mut cache_len = None;
        for (layer, kv) in self.kvs.iter().enumerate() {
            let Some((k, v)) = kv else { continue };
            let k_len = k.dim(2)?;
            let v_len = v.dim(2)?;
            if k_len != v_len {
                bail!("invalid KV cache at layer {layer}: K has {k_len} tokens but V has {v_len}")
            }
            match cache_len {
                Some(expected) if k_len != expected => bail!(
                    "invalid KV cache at layer {layer}: expected {expected} tokens, got {k_len}"
                ),
                None => cache_len = Some(k_len),
                _ => {}
            }
        }
        Ok(cache_len.unwrap_or(0))
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    fn validate_forward(&self, index_pos: usize, seq_len: usize) -> Result<usize> {
        let end_pos = index_pos
            .checked_add(seq_len)
            .ok_or_else(|| candle_core::Error::Msg("token position overflow".to_string()))?;
        if end_pos > self.max_position_embeddings {
            bail!(
                "context length {end_pos} exceeds max_position_embeddings {}",
                self.max_position_embeddings
            )
        }
        let cache_len = self.len()?;
        if self.use_kv_cache && cache_len != index_pos {
            bail!(
                "KV cache/index mismatch: cache contains {cache_len} tokens but index_pos is {index_pos}"
            )
        }
        Ok(if self.use_kv_cache { cache_len } else { 0 })
    }

    fn mask(&mut self, seq_len: usize, past_len: usize) -> Result<Tensor> {
        let kv_len = past_len + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask =
                candle_transformers::utils::build_causal_mask(seq_len, past_len, &self.device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }
}

impl GraniteConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        let num_key_value_heads = self.num_key_value_heads();
        Config {
            architectures: self.architectures,
            attention_bias: self.attention_bias,
            attention_dropout: self.attention_dropout,
            attention_multiplier: self.attention_multiplier,
            bos_token_id: self.bos_token_id,
            embedding_multiplier: self.embedding_multiplier,
            eos_token_id: self.eos_token_id,
            hidden_act: self.hidden_act,
            hidden_size: self.hidden_size,
            initializer_range: self.initializer_range,
            intermediate_size: self.intermediate_size,
            logits_scaling: self.logits_scaling,
            max_position_embeddings: self.max_position_embeddings,
            mlp_bias: self.mlp_bias,
            model_type: self.model_type,
            num_attention_heads: self.num_attention_heads,
            num_hidden_layers: self.num_hidden_layers,
            num_key_value_heads,
            pad_token_id: self.pad_token_id,
            residual_multiplier: self.residual_multiplier,
            rms_norm_eps: self.rms_norm_eps,
            rope_scaling: self.rope_scaling,
            rope_theta: self.rope_theta,
            tie_word_embeddings: self.tie_word_embeddings,
            torch_dtype: self.torch_dtype,
            transformers_version: self.transformers_version,
            use_cache: self.use_cache,
            vocab_size: self.vocab_size,
            use_flash_attn,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub attention_multiplier: f32,
    pub bos_token_id: Option<u32>,
    pub embedding_multiplier: f32,
    pub eos_token_id: Option<GraniteEosToks>,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub logits_scaling: f32,
    pub max_position_embeddings: usize,
    pub mlp_bias: bool,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub pad_token_id: Option<u32>,
    pub residual_multiplier: f32,
    pub rms_norm_eps: f64,
    pub rope_scaling: Option<GraniteRopeConfig>,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: usize,
    pub use_flash_attn: bool,
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
    max_position_embeddings: usize,
    attention_multiplier: f32,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

impl CausalSelfAttention {
    fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
        let shape = mask.shape();
        let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
        let m = mask.where_cond(&on_true, on_false)?;
        Ok(m)
    }
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }
    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        candle_transformers::utils::repeat_kv(
            x,
            self.num_attention_heads / self.num_key_value_heads,
        )
    }
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
        past_len: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
            }
            let k_seq_len = k.dim(2)?;
            let v_seq_len = v.dim(2)?;
            if k_seq_len != v_seq_len {
                bail!("K/V sequence length mismatch: K={k_seq_len}, V={v_seq_len}")
            }
            if k_seq_len > self.max_position_embeddings {
                bail!(
                    "KV cache length {k_seq_len} exceeds max_position_embeddings {}",
                    self.max_position_embeddings
                )
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()))
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = self.attention_multiplier;
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? * self.attention_multiplier as f64)?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache.mask(seq_len, past_len)?.broadcast_as(att.shape())?;
                Self::masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("attn_q"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("attn_k"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("attn_v"))?;
        let o_proj = linear(size_q, size_in, vb.pp("attn_output"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            span,
            span_rot,
            max_position_embeddings: cfg.max_position_embeddings,
            attention_multiplier: cfg.attention_multiplier,
        })
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}
impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("ffn_gate"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("ffn_up"))?;
        let c_proj = linear(i_size, h_size, vb.pp("ffn_down"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
    residual_multiplier: f32,
}
impl Block {
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
        past_len: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = ((self
            .attn
            .forward(&x, index_pos, block_idx, cache, past_len)?
            * self.residual_multiplier as f64)?
            + residual)?;
        let residual = &x;
        let x = ((self.mlp.forward(&self.rms_2.forward(&x)?)? * self.residual_multiplier as f64)?
            + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "blk");
        let attn = CausalSelfAttention::load(vb.clone(), cfg)?;
        let mlp = Mlp::load(vb.clone(), cfg)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("attn_norm"))?;
        let rms_2 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("ffn_norm"))?;

        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
            residual_multiplier: cfg.residual_multiplier,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Granite {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    embedding_multiplier: f32,
    logits_scaling: f32,
}

impl Granite {
    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        if seq_len == 0 {
            bail!("cannot run Granite with an empty token sequence")
        }
        let past_len = cache.validate_forward(index_pos, seq_len)?;
        let mut x = self.wte.forward(x)?;
        x = (x * self.embedding_multiplier as f64)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache, past_len)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)? / self.logits_scaling as f64
    }

    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wte = Embedding::new(cfg.vocab_size, cfg.hidden_size, vb.pp("token_embd"))?;
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("output_norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("blk.{i}")), cfg))
            .collect::<Result<_>>()?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("token_embd"))?;
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            embedding_multiplier: cfg.embedding_multiplier,
            logits_scaling: cfg.logits_scaling,
        })
    }

    #[cfg(test)]
    pub(crate) fn forward_hidden_trace(
        &self,
        input: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Vec<Tensor>> {
        let (_batch_size, seq_len) = input.dims2()?;
        let past_len = cache.validate_forward(index_pos, seq_len)?;
        let mut hidden = (self.wte.forward(input)? * self.embedding_multiplier as f64)?;
        let mut trace = vec![hidden.i((.., seq_len - 1, ..))?.contiguous()?];

        for (block_idx, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, index_pos, block_idx, cache, past_len)?;
            trace.push(hidden.i((.., seq_len - 1, ..))?.contiguous()?);
        }
        trace.push(
            self.ln_f
                .forward(&hidden)?
                .i((.., seq_len - 1, ..))?
                .contiguous()?,
        );
        Ok(trace)
    }

    #[cfg(test)]
    pub(crate) fn forward_first_block_trace(
        &self,
        input: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Vec<(&'static str, Tensor)>> {
        let (_batch_size, seq_len) = input.dims2()?;
        let past_len = cache.validate_forward(index_pos, seq_len)?;
        let block = &self.blocks[0];
        let hidden = (self.wte.forward(input)? * self.embedding_multiplier as f64)?;
        let normalized = block.rms_1.forward(&hidden)?;

        let q_raw = block.attn.q_proj.forward(&normalized)?;
        let k_raw = block.attn.k_proj.forward(&normalized)?;
        let v_raw = block.attn.v_proj.forward(&normalized)?;
        let (batch_size, _, _hidden_size) = normalized.dims3()?;
        let q = q_raw
            .reshape((
                batch_size,
                seq_len,
                block.attn.num_attention_heads,
                block.attn.head_dim,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k_raw
            .reshape((
                batch_size,
                seq_len,
                block.attn.num_key_value_heads,
                block.attn.head_dim,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let q_rope = block.attn.apply_rotary_emb(&q, index_pos, cache)?;
        let k_rope = block.attn.apply_rotary_emb(&k, index_pos, cache)?;
        let attention = block
            .attn
            .forward(&normalized, index_pos, 0, cache, past_len)?;
        let after_attention = ((&attention * block.residual_multiplier as f64)? + &hidden)?;
        let mlp_input = block.rms_2.forward(&after_attention)?;
        let mlp_output = block.mlp.forward(&mlp_input)?;
        let block_output = ((&mlp_output * block.residual_multiplier as f64)? + &after_attention)?;

        let last_seq = |tensor: &Tensor| tensor.i((.., seq_len - 1, ..))?.contiguous();
        let last_heads = |tensor: &Tensor| {
            tensor
                .i((.., .., seq_len - 1, ..))?
                .contiguous()?
                .reshape((batch_size, tensor.dim(1)? * block.attn.head_dim))
        };

        Ok(vec![
            ("embedding", last_seq(&hidden)?),
            ("rms_1", last_seq(&normalized)?),
            ("q_projection", last_seq(&q_raw)?),
            ("k_projection", last_seq(&k_raw)?),
            ("v_projection", last_seq(&v_raw)?),
            ("q_rope", last_heads(&q_rope)?),
            ("k_rope", last_heads(&k_rope)?),
            ("attention_output", last_seq(&attention)?),
            ("after_attention_residual", last_seq(&after_attention)?),
            ("rms_2", last_seq(&mlp_input)?),
            ("mlp_output", last_seq(&mlp_output)?),
            ("block_output", last_seq(&block_output)?),
        ])
    }

    #[cfg(test)]
    pub(crate) fn first_block_projection_shape_diffs(
        &self,
        token_id: u32,
        repeats: usize,
        device: &Device,
    ) -> Result<Vec<(&'static str, f32)>> {
        let input = Tensor::new(&[token_id], device)?.unsqueeze(0)?;
        let hidden = (self.wte.forward(&input)? * self.embedding_multiplier as f64)?;
        let normalized = self.blocks[0].rms_1.forward(&hidden)?;
        let repeated = Tensor::cat(&vec![&normalized; repeats], 1)?;
        let block = &self.blocks[0];

        let compare = |projection: &Linear| -> Result<f32> {
            let single = projection.forward(&normalized)?;
            let batched = projection.forward(&repeated)?;
            let batched_last = batched.i((.., repeats - 1, ..))?.unsqueeze(1)?;
            (&single - &batched_last)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()
        };

        Ok(vec![
            ("q_projection", compare(&block.attn.q_proj)?),
            ("k_projection", compare(&block.attn.k_proj)?),
            ("v_projection", compare(&block.attn.v_proj)?),
            ("o_projection", compare(&block.attn.o_proj)?),
            ("mlp_gate", compare(&block.mlp.c_fc1)?),
            ("mlp_up", compare(&block.mlp.c_fc2)?),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_with_kv(
        k_shape: (usize, usize, usize, usize),
        v_shape: (usize, usize, usize, usize),
    ) -> Cache {
        let device = Device::Cpu;
        Cache {
            masks: HashMap::new(),
            use_kv_cache: true,
            kvs: vec![Some((
                Tensor::zeros(k_shape, DType::F32, &device).unwrap(),
                Tensor::zeros(v_shape, DType::F32, &device).unwrap(),
            ))],
            cos: Tensor::zeros((16, 2), DType::F32, &device).unwrap(),
            sin: Tensor::zeros((16, 2), DType::F32, &device).unwrap(),
            device,
            max_position_embeddings: 16,
        }
    }

    #[test]
    fn cache_len_uses_the_sequence_axis() {
        let cache = cache_with_kv((1, 2, 7, 4), (1, 2, 7, 4));
        assert_eq!(cache.len().unwrap(), 7);
    }

    #[test]
    fn cache_rejects_different_k_and_v_lengths() {
        let cache = cache_with_kv((1, 2, 7, 4), (1, 2, 6, 4));
        assert!(cache.len().is_err());
    }

    #[test]
    fn forward_position_must_match_cache_length() {
        let cache = cache_with_kv((1, 2, 7, 4), (1, 2, 7, 4));
        assert!(cache.validate_forward(6, 1).is_err());
        assert_eq!(cache.validate_forward(7, 1).unwrap(), 7);
    }

    #[test]
    fn forward_rejects_positions_past_rope_table() {
        let cache = cache_with_kv((1, 2, 15, 4), (1, 2, 15, 4));
        assert!(cache.validate_forward(15, 2).is_err());
    }
}
