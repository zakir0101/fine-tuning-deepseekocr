# !/usr/bin/env python3

import sys
import io
import json
import fitz
from PIL import Image, ImageOps
import re
import os
from lora_inference_vllm import load_lora_model, run_ocr_inference
from utils import (
    IGCSE_HOME,
    MODEL_NAME,
    MODEL_NAME_SUFFIX,
    MODEL_OUTPUT_DIR,
    PROMPT,
)


MULTIPLE_CHOICE_PAPERS = ["9702_1", "0625_1", "0625_2"]


def main():

    llm, sampling_param = load_lora_model(
        model_name=f"zakir0101/{MODEL_NAME}{MODEL_NAME_SUFFIX}"
    )
    # igcse_dir = "/home/zakir/IGCSE_DATA"
    igcse_dir = IGCSE_HOME
    # output_igcse_dir = igcse_dir
    output_igcse_dir = "/content/IGCSE_DATA"

    sub_count = 0
    for subject in os.listdir(igcse_dir):
        if not re.match(r"\d{4}", subject):
            continue

        for pdf in os.listdir(os.path.join(igcse_dir, subject, "exams")):

            path_pdf = os.path.join(igcse_dir, subject, "exams", pdf)
            exam_name = pdf.split(".")[0]
            year = int(exam_name.split("_")[1][1:])
            if year != 25:
                print("skippning year ", year)
                continue

            paper_name = exam_name.split("_")[-1][0]
            subject_paper_id = subject + "_" + paper_name

            prompt = PROMPT
            print("*************************************************")
            print(f"******************{exam_name}********************")
            out_filename = "v1.json"

            output_dir = os.path.join(
                output_igcse_dir,
                subject,
                MODEL_OUTPUT_DIR,
                exam_name,
            )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, out_filename)
            output_path_source = os.path.join(
                igcse_dir, subject, MODEL_OUTPUT_DIR, exam_name, out_filename
            )
            if os.path.exists(output_path) or os.path.exists(
                output_path_source
            ):
                print("[-] skipping ", exam_name, "already [Done]")
                continue

            doc = fitz.open(path_pdf)
            out = []
            counter = 0
            for i in range(0, len(doc), 30):
                counter += 1
                mat = fitz.Matrix(4, 4)
                # pix = page_doc.get_pixmap(matrix=mat)
                # img = Image.frombytes(
                #     "RGB", [pix.width, pix.height], pix.samples
                # )
                images = []
                for j in range(2):

                    if (i + j) >= len(doc):
                        continue
                    page_doc = doc[i + j]
                    pixmap = page_doc.get_pixmap(matrix=mat, alpha=False)
                    Image.MAX_IMAGE_PIXELS = None

                    img_data = pixmap.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))

                    # Convert to RGB like official code
                    if img.mode in ("RGBA", "LA"):
                        background = Image.new(
                            "RGB", img.size, (255, 255, 255)
                        )
                        background.paste(
                            img,
                            mask=(
                                img.split()[-1] if img.mode == "RGBA" else None
                            ),
                        )
                        img = background
                    elif img.mode != "RGB":
                        img = img.convert("RGB")

                    images.append(img)
                # Optional parameters
                print(f"### Processing {len(images)} Images :")

                # (
                #     sys.argv[3] if len(sys.argv) > 3 else "<image>\nFree OCR. "
                # )

                # image_size = int(sys.argv[4]) if len(sys.argv) > 4 else 640
                # base_size = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
                try:

                    # Run inference
                    model_outputs = run_ocr_inference(
                        llm,
                        sampling_param,
                        images,
                        prompt=prompt,
                    )
                    for j, output in enumerate(model_outputs):
                        out.append(
                            {
                                "page": (i + 1 + j),
                                "raw_output": output.outputs[0].text,
                            }
                        )

                except Exception as e:
                    print(f"Error: {str(e)}", file=sys.stderr)
                    import traceback

                    traceback.print_exc(file=sys.stderr)
                    sys.exit(1)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps({"pages": out}, indent=4, ensure_ascii=False)
                )
                print("successfully saved", output_path)


if __name__ == "__main__":
    main()
