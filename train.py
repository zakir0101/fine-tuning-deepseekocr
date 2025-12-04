import os
import tempfile


from unsloth import is_bf16_supported, FastVisionModel

import json
from pathlib import Path
import sys
import fitz
import torch
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
import io
from transformers import Trainer, TrainingArguments, AutoModel
from collator import DeepSeekOCRDataCollator
from transformers import TrainerCallback

# from deepseek_ocr.modeling_deepseekocr import (
#     format_messages,
#     text_encode,
#     BasicImageTransform,
#     dynamic_preprocess,
# )
from utils import (
    BASE_MODEL_PATH,
    LOG_DIR,
    MODEL_NAME,
    OUTPUT_DIR,
    load_training_dataset,
)


os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
local_rank = int(os.environ.get("LOCAL_RANK", -1))

model, tokenizer = FastVisionModel.from_pretrained(
    BASE_MODEL_PATH,
    load_in_4bit=False,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    # CRITICAL: Map the model to the specific GPU for this process
    device_map={"": f"cuda:{local_rank}"},
    auto_model=AutoModel,
    trust_remote_code=True,
    unsloth_force_compile=True,
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    r=16,  # The larger, the higher the accuracy, but might overfit
    lora_alpha=16,  # Recommended alpha == r at least
    lora_dropout=0.05,  # was 0
    bias="none",
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


# root = Path("/home/zakir/IGCSE_DATA")
converted_dataset, validation_dataset = load_training_dataset()


print("TRAINING DATA length = ", len(converted_dataset))
print("VALIDATION DATA length = ", len(validation_dataset))

FastVisionModel.for_training(model)  # Enable for training!


data_collator = DeepSeekOCRDataCollator(
    tokenizer=tokenizer,
    model=model,
    image_size=640,
    base_size=1024,
    crop_mode=True,
    train_on_responses_only=True,
)


class SanityCheckCallback(TrainerCallback):
    """
    Forces an evaluation at specific early steps to ensure the
    validation loop doesn't crash, regardless of eval_delay.
    """

    def on_step_end(self, args, state, control, **kwargs):
        # Trigger eval at Step 20 just to check for Crashes/OOM
        if state.global_step == 20:
            print("\n🚨 TRIGGERING EARLY SANITY CHECK EVALUATION (Step 20) 🚨")
            control.should_evaluate = True
        return control


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,  # Must use!
    # train_dataset=converted_dataset,
    train_dataset=converted_dataset,  # <--- CHANGED: Uses only the training split
    eval_dataset=validation_dataset,  # <--- ADDED: Uses the quarantined real data
    callbacks=[SanityCheckCallback()],
    args=TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=6,
        warmup_steps=5,
        # max_steps=750,  # 60
        num_train_epochs=1,  # Set this instead of max_steps for full training runs
        learning_rate=2e-4,
        optim="adamw_8bit",
        weight_decay=0.05,  # 0.001
        lr_scheduler_type="linear",
        seed=3407,
        fp16=not is_bf16_supported(),  # Use fp16 if bf16 is not supported
        bf16=is_bf16_supported(),  # Use bf16 if supported
        output_dir=OUTPUT_DIR,
        # report_to="none",  # For Weights and Biases
        report_to="tensorboard" if local_rank == 0 else "none",
        disable_tqdm=True if local_rank != 0 else False,
        # DDP SETTINGS:
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8,
        # You MUST put the below items for vision finetuning:
        remove_unused_columns=False,
        # --- NEW SETTINGS ADDED BELOW ---
        save_strategy="steps",  # Required for load_best_model_at_end
        save_steps=200,  # Save every 200 steps (matches plan)
        eval_strategy="steps",  # Evaluate every 200 steps
        eval_steps=200,  # Matches save_steps
        eval_delay=600,
        save_total_limit=3,  # <--- ADDED: Only keep top 3 checkpoints
        load_best_model_at_end=True,  # <--- ADDED: Auto-load the best checkpoint after training
        metric_for_best_model="loss",  # Metric to track
        greater_is_better=False,  # Lower loss is better
        logging_dir=LOG_DIR,
        logging_strategy="steps",
        logging_steps=20,
    ),
)


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(
    torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# ********************************************
trainer_stats = trainer.train()


# used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# used_percentage = round(used_memory / max_memory * 100, 3)
# lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
# print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
# print(
#     f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
# )
# print(f"Peak reserved memory = {used_memory} GB.")
# print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
# print(f"Peak reserved memory % of max memory = {used_percentage} %.")
# print(
#     f"Peak reserved memory for training % of max memory = {lora_percentage} %."
# )

mname = MODEL_NAME


model.push_to_hub(
    f"zakir0101/{mname}", token="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"
)  # Online saving
tokenizer.push_to_hub(
    f"zakir0101/{mname}", token="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"
)  # Online saving
model.push_to_hub_merged(
    f"zakir0101/{mname}_b16",
    tokenizer,
    token="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",
)

model.save_pretrained(mname)  # Local saving
tokenizer.save_pretrained(mname)
