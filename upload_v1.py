from huggingface_hub import HfApi
from utils import MODEL_NAME

# Just upload the LoRA adapter directly
api = HfApi()

mname = MODEL_NAME


repo_id = f"zakir0101/{mname}_lora"

# CREATE THE REPO FIRST
print(f"Creating repository {repo_id}...")
api.create_repo(
    repo_id=repo_id,
    token="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",
    repo_type="model",
    exist_ok=True,  # Won't error if repo already exists
)


print("Uploading LoRA checkpoint...")
api.upload_folder(
    folder_path="outputs/checkpoint-1200",
    repo_id=repo_id,  # Note: _lora not _b16
    token="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",
    repo_type="model",
)

print("✅ LoRA adapter uploaded!")
print(f"Users can load it with:")
print(
    f'model = PeftModel.from_pretrained(base_model, "zakir0101/{mname}_lora")'
)
