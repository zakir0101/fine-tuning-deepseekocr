import os
import fitz
from torch.utils.data import Dataset
import json
from pathlib import Path
from PIL import Image
import random

is_local = False
## TRAINing CONFIG --------------------------


MULTIPLE_CHOICE_PAPERS = ["9702_1", "0625_1", "0625_2"]

MODEL_NAME = "ftv5-ocr"
MODEL_NAME_SUFFIX = "_b16"
MODEL_OUTPUT_DIR = "ft5-deepseekocr"
# PROMPT = "<image>\n<|grounding|>Convert the document to markdown. extract the questions elements "
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

PROMPT_MULTI_CHOICE = "<image>\n<|grounding|>Convert the document to markdown. and extract the structural elements for each MULTI-CHOICE question"

SYNTHETIC_FILENAME = "synthetic_v1_ft5-deepseek.json"
REAL_FILENAME = "mathpix_v1_pdfext_v1_ft5-deepseek.json"

# NOTE: ******************* DIR ****************

IGCSE_HOME = "./IGCSE_DATA"
BASE_MODEL_PATH = "./deepseek_ocr"
OUTPUT_DIR = "./outputs"
LOG_DIR = "./logs"
# IGCSE_HOME = "/kaggle/input/igcse-dataset/IGCSE_DATA"
# BASE_MODEL_PATH = "/kaggle/input/deepseekocr/transformers/unsloth/1/deepseek_ocr"
# OUTPUT_DIR = "/kaggle/working/outputs"
# LOG_DIR = "/kaggle/working/logs"

igcse_root = Path(IGCSE_HOME)

## Download CONFIG --------------------------


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


def convert_to_conversation(image_pil, raw_output, instruction):
    """Convert dataset sample to conversation format"""
    conversation = [
        {
            "role": "<|User|>",
            "content": instruction,
            "images": [image_pil],
        },
        {"role": "<|Assistant|>", "content": raw_output},
    ]
    return {"messages": conversation}


# def get_image(doc, page_number: int):
#     page = doc[page_number - 1]
#     mat = fitz.Matrix(2, 2)
#     pix = page.get_pixmap(matrix=mat)
#     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#     return img


def add_training_data_to_list(converted_dataset, p):
    subject_name, exam_name = (parts := p.absolute().parts)[-4], parts[-2]
    pdf_path = Path(IGCSE_HOME, subject_name, "exams", f"{exam_name}.pdf")
    paper_name = exam_name.split("_")[-1][0]
    subject_paper_id = subject_name + "_" + paper_name
    if subject_paper_id in MULTIPLE_CHOICE_PAPERS:
        print("found multi choice exam")
        print("[-] SKIPPING")
        return

    if not os.path.exists(pdf_path):
        print("PDF_file not FOUND !!", pdf_path)
        return

    doc = fitz.open(pdf_path)

    with p.open("r", encoding="utf-8") as f:
        json_data: dict = json.loads(f.read())

    for content in json_data.values():
        page_number = content["page"]
        if page_number > len(doc):
            break
        raw_output = content["raw_output"]
        meta_entry = {
            "pdf_path": pdf_path,
            "page": page_number,
            "raw_output": raw_output,
        }
        converted_dataset.append(meta_entry)
    doc.close()


def load_training_dataset():

    json_files = list(
        igcse_root.glob(f"synthetic/training/*/{SYNTHETIC_FILENAME}")
    )
    metadata_list = []
    for p in json_files:
        add_training_data_to_list(metadata_list, p)

    print(f"Indexed {len(metadata_list)} samples.")

    random.shuffle(metadata_list)
    meta_valid = metadata_list[:900]
    meta_train = metadata_list[900:]

    converted_dataset = DeepSeekOCRLazyDataset(meta_train, PROMPT)
    validation_dataset = DeepSeekOCRLazyDataset(meta_valid, PROMPT)
    return converted_dataset, validation_dataset


if __name__ == "__main__":
    train_d, valid_d = load_training_dataset()
    print("Training length = ", len(train_d))
    print("validation length = ", len(valid_d))
