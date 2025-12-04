from huggingface_hub import snapshot_download
from utils import BASE_MODEL_PATH

snapshot_download("unsloth/DeepSeek-OCR", local_dir=BASE_MODEL_PATH)
