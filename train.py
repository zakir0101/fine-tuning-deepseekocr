import os
from unsloth import is_bf16_supported, FastVisionModel
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoModel
from collator import DeepSeekOCRDataCollator
from transformers import TrainerCallback

from utils import (
    BASE_MODEL_PATH,
    LOG_DIR,
    MODEL_NAME,
    OUTPUT_DIR,
    PROMPT,
    load_training_dataset,
)


os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"


class DeepSeekOCRLazyDataset(Dataset):
    def __init__(self, metadata_list, prompt_instruction):
        """
        metadata_list: List of dicts containing {'pdf_path': str, 'page': int, 'raw_output': str}
        """
        self.metadata = metadata_list
        self.instruction = prompt_instruction

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        try:
            doc = fitz.open(item["pdf_path"])
            page = doc[item["page"] - 1]
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            image = Image.frombytes(
                "RGB", [pix.width, pix.height], pix.samples
            )
            doc.close()  # CRITICAL: Close file handle
        except Exception as e:
            print(f"Error loading {item['pdf_path']} page {item['page']}: {e}")
            # Return a black dummy image to prevent crash, or raise error
            image = Image.new("RGB", (640, 640), color="black")

        convo = convert_to_conversation(
            image, item["raw_output"], self.instruction
        )

        return convo


model, tokenizer = FastVisionModel.from_pretrained(
    BASE_MODEL_PATH,
    load_in_4bit=False,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    auto_model=AutoModel,
    trust_remote_code=True,
    unsloth_force_compile=True,
    use_gradient_checkpointing="unsloth",  # "unsloth",  # True or "unsloth" for long context
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


meta_train, meta_valid = load_training_dataset()
converted_dataset = DeepSeekOCRLazyDataset(meta_train, PROMPT)
validation_dataset = DeepSeekOCRLazyDataset(meta_valid, PROMPT)

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
        if state.global_step == 10:
            print("\n🚨 TRIGGERING EARLY SANITY CHECK EVALUATION (Step 20) 🚨")
            control.should_evaluate = True
        return control


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=converted_dataset,
    eval_dataset=validation_dataset,
    callbacks=[SanityCheckCallback()],
    args=TrainingArguments(
        per_device_train_batch_size=10,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
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
        report_to="tensorboard",
        dataloader_num_workers=20,
        # You MUST put the below items for vision finetuning:
        remove_unused_columns=False,
        # --- NEW SETTINGS ADDED BELOW ---
        save_strategy="steps",  # Required for load_best_model_at_end
        save_steps=200,  # Save every 200 steps (matches plan)
        eval_strategy="steps",  # Evaluate every 200 steps
        eval_steps=200,  # Matches save_steps
        eval_delay=600,
        save_total_limit=1,  # <--- ADDED: Only keep top 3 checkpoints
        load_best_model_at_end=True,  # <--- ADDED: Auto-load the best checkpoint after training
        metric_for_best_model="loss",  # Metric to track
        greater_is_better=False,  # Lower loss is better
        logging_dir=LOG_DIR,
        logging_strategy="steps",
        logging_steps=5,
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
# 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
model.save_pretrained(mname)  # Local saving
tokenizer.save_pretrained(mname)
