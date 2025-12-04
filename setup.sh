    # Do this only in Colab notebooks! Otherwise use pip install unsloth
# import torch; v = re.match(r"[0-9\.]{3,}", str(torch.__version__)).group(0)
# xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")
pip install --no-deps bitsandbytes accelerate  peft trl triton cut_cross_entropy unsloth_zoo
pip install --no-deps xformers==0.0.32.post2
pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
pip install --no-deps unsloth
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2
pip install jiwer
pip install einops addict easydict
pip install  torch==2.8.0 psutil torchvision torchao diffusers tyro wheel  msgspec  matplotlib

pip install pillow PyMuPDF
