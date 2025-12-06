import sys
import io
import json
import fitz
from PIL import Image
import re
import os
import torch
import multiprocessing

# from lora_inference_vllm import load_lora_model, run_ocr_inference
from utils import (
    IGCSE_HOME,
    MODEL_NAME,
    MODEL_NAME_SUFFIX,
    MODEL_OUTPUT_DIR,
    PROMPT,
)

# Constants
MULTIPLE_CHOICE_PAPERS = ["9702_1", "0625_1", "0625_2"]
IGCSE_DIR = IGCSE_HOME
OUTPUT_IGCSE_DIR = "/content/IGCSE_DATA"


def get_all_tasks():
    """
    Scans directories and returns a list of all PDF tasks to process.
    """
    tasks = []

    if not os.path.exists(IGCSE_DIR):
        print(f"Warning: Directory {IGCSE_DIR} not found.")
        return []

    for subject in os.listdir(IGCSE_DIR):
        if not re.match(r"\d{4}", subject):
            continue

        exams_path = os.path.join(IGCSE_DIR, subject, "exams")
        if not os.path.exists(exams_path):
            continue

        for pdf in os.listdir(exams_path):
            path_pdf = os.path.join(exams_path, pdf)
            exam_name = pdf.split(".")[0]

            # Filter logic (Year check)
            try:
                year = int(exam_name.split("_")[1][1:])
                if year != 25:
                    continue
            except:
                continue

            # Output check
            out_filename = "v1.json"
            output_dir = os.path.join(
                OUTPUT_IGCSE_DIR, subject, MODEL_OUTPUT_DIR, exam_name
            )
            output_path = os.path.join(output_dir, out_filename)

            # Check if source already has output
            output_path_source = os.path.join(
                IGCSE_DIR, subject, MODEL_OUTPUT_DIR, exam_name, out_filename
            )

            if os.path.exists(output_path) or os.path.exists(
                output_path_source
            ):
                continue

            # Add task to list
            tasks.append(
                {
                    "subject": subject,
                    "exam_name": exam_name,
                    "path_pdf": path_pdf,
                    "output_dir": output_dir,
                    "output_path": output_path,
                }
            )
    return tasks


def process_batch(gpu_id, tasks_subset):
    """
    Worker function that initializes a vLLM engine on a specific GPU
    and processes the assigned subset of tasks.
    """
    print(
        f"[GPU {gpu_id}] Starting worker. Assigned {len(tasks_subset)} exams."
    )

    # --- CRITICAL: ISOLATE GPU ---
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import inside process to avoid VLLM context conflicts
    from lora_inference_vllm import load_lora_model, run_ocr_inference

    # Revert your lora_inference_vllm.py to use tensor_parallel_size=1
    # We are handling parallelism manually here.
    try:
        model_full_name = f"zakir0101/{MODEL_NAME}{MODEL_NAME_SUFFIX}"
        llm, sampling_param = load_lora_model(model_name=model_full_name)
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to load model: {e}")
        return

    prompt = PROMPT

    for task in tasks_subset:
        exam_name = task["exam_name"]
        path_pdf = task["path_pdf"]
        output_path = task["output_path"]
        output_dir = task["output_dir"]

        print(f"[GPU {gpu_id}] Processing: {exam_name}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            doc = fitz.open(path_pdf)
            out = []

            # Process in batches
            BATCH_SIZE = 30

            for i in range(0, len(doc), BATCH_SIZE):
                mat = fitz.Matrix(4, 4)
                images = []

                # Load Batch of Images
                for j in range(BATCH_SIZE):
                    if (i + j) >= len(doc):
                        break

                    page_doc = doc[i + j]
                    pixmap = page_doc.get_pixmap(matrix=mat, alpha=False)
                    img_data = pixmap.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))

                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    images.append(img)

                if not images:
                    continue

                # Run Inference
                try:
                    model_outputs = run_ocr_inference(
                        llm, sampling_param, images, prompt=prompt
                    )
                    for j, output in enumerate(model_outputs):
                        out.append(
                            {
                                "page": (i + 1 + j),
                                "raw_output": output.outputs[0].text,
                            }
                        )
                except Exception as e:
                    print(f"[GPU {gpu_id}] Error in batch inference: {e}")

            # Save Results
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps({"pages": out}, indent=4, ensure_ascii=False)
                )
            print(f"[GPU {gpu_id}] Saved: {exam_name}")

        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing file {exam_name}: {e}")


def main():
    # 1. Gather all work
    all_tasks = get_all_tasks()
    print(f"Total exams to process: {len(all_tasks)}")

    if len(all_tasks) == 0:
        print("No tasks found.")
        return

    # 2. Split work into two chunks
    mid_point = len(all_tasks) // 2
    tasks_gpu0 = all_tasks[:mid_point]
    tasks_gpu1 = all_tasks[mid_point:]

    # 3. Create Processes
    p1 = multiprocessing.Process(target=process_batch, args=(0, tasks_gpu0))
    p2 = multiprocessing.Process(target=process_batch, args=(1, tasks_gpu1))

    # 4. Start Processes
    p1.start()
    p2.start()

    # 5. Wait for completion
    p1.join()
    p2.join()
    print("All tasks completed.")


if __name__ == "__main__":
    # Ensure spawn method is used (safer for CUDA + Torch)
    multiprocessing.set_start_method("spawn", force=True)
    main()
