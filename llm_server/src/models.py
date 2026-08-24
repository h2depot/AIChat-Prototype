import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
import accelerate

MODEL_ID = "ibm-granite/granite-4.1-3b"


class LLM:
    def __init__(self):
        self.model_name = MODEL_ID
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            dtype=dtype,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 2048) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ).strip()
