from unsloth import FastVisionModel
import logging
from transformers import Trainer, TrainingArguments, AutoModel
from utils import MODEL_NAME


# --- STEP 1: THE FIX (Silence the paranoid logger) ---
# Unsloth listens to this specific logger to decide if it should crash.
# We set it to ERROR so it ignores warnings about "position_ids".
logger = logging.getLogger("transformers.modeling_utils")
logger.setLevel(logging.ERROR)

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="outputs/checkpoint-1000",  # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit=False,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    auto_model=AutoModel,
    trust_remote_code=True,
    unsloth_force_compile=True,
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)


mname = MODEL_NAME + "_1000"

model.push_to_hub_merged(
    f"zakir0101/{mname}_b16",
    tokenizer,
    token="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",
)
