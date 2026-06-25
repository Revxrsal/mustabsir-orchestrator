import sys
print("1. Starting...", flush=True)

import torch
print(f"2. Torch version: {torch.__version__}", flush=True)

print("3. Attempting to import BGEM3FlagModel from FlagEmbedding...", flush=True)
from FlagEmbedding import BGEM3FlagModel
print("4. BGEM3FlagModel imported successfully!", flush=True)

print("5. Initializing model weights...", flush=True)
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices="cuda")
print("6. Model loaded successfully!", flush=True)
