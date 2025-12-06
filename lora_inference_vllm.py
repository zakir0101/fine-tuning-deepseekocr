import os
import sys
import builtins
from utils import MODEL_NAME, MODEL_NAME_SUFFIX, PROMPT
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

# Suppress Unsloth warning about uninitialized weights (position_ids is normal)
# os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

# Monkey-patch input() to automatically return 'y' for transformers prompts
_original_input = builtins.input


def _auto_yes_input(prompt=""):
    prompt_lower = prompt.lower()
    if any(
        keyword in prompt_lower
        for keyword in [
            "run the custom code",
            "trust_remote_code",
            "remote code execution",
            "[y/n]",
        ]
    ):
        print("y")  # Print 'y' so it looks like user responded
        return "y"
    return _original_input(prompt)


builtins.input = _auto_yes_input


def load_lora_model(model_name):
    """
    Load the LoRA fine-tuned DeepSeek OCR model.

    Args:
        model_name: Path to the LoRA model directory (default: "lora_model")
        load_in_4bit: Whether to use 4bit quantization (default: False)

    Returns:
        tuple: (model, tokenizer)
    """

    llm = LLM(
        model=model_name,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
        trust_remote_code=True,
        tensor_parallel_size=2,  # <--- THIS IS THE KEY CHANGE
    )
    sampling_param = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        # ngram logit processor args
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
        ),
        skip_special_tokens=False,
    )
    return llm, sampling_param


def run_ocr_inference(llm, sampling_param, images_list, prompt):
    model_input = []
    for img in images_list:
        model_input.append(
            {"prompt": prompt, "multi_modal_data": {"image": img}},
        )

    model_outputs = llm.generate(model_input, sampling_param)
    return model_outputs


# Example usage (for testing)
if __name__ == "__main__":
    llm, sampling_param = load_lora_model(
        f"zakir0101/{MODEL_NAME}{MODEL_NAME_SUFFIX}"
    )

    prompt = PROMPT
    image_file = "test-image.png"
    img = Image.open(image_file).convert("RGB")

    model_outputs = run_ocr_inference(llm, sampling_param, [img], prompt)
    for output in model_outputs:
        print(output.outputs[0].text)
