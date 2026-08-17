//! Granite is a Long Context Transformer Language Model.
//!
//! A high performance transformer model optimized for efficient processing
//! of very long context sequences
//! 
//! Excuse Me!!! This original code is from "https://github.com/huggingface/candle.git"
//! I have modified it to fit my needs, but I want to give credit to the original authors for their work and contributions to the open-source community.
//! Thank you very much!!

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_transformers::{
    quantized_nn::{linear_no_bias as linear, Embedding, Linear, RmsNorm},
    quantized_var_builder::VarBuilder,
};
use std::{collections::HashMap, f32::consts::PI};

