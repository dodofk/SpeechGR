"""
This code is used to train subword modeling (bpe) with discrete token from hu-bert

Assume code is extract and save in .h5 file format.

Config directly at the top of the code.

"""
import os
import glob
import h5py
import sentencepiece as spm
import tempfile
import numpy as np
from tqdm import tqdm

VOCAB_SIZE = 2000
code_dir = "/home/ricky/dodofk/dataset/slue_sqa_code_l22_c1000"
code_files = glob.glob(os.path.join(code_dir, "*/*.code"))
transform_code_dir = "/home/ricky/dodofk/dataset/slue_sqa_code_l22_c1000_bpe"
model_prefix = "bpe_model/slue_sqa_code_l22_c1000"

split_code_name = [
    "train_code",
    "validation_code",
    "test_code",
    "verified_test_code",
    "document_code"
]

if not os.path.exists(transform_code_dir):
    os.makedirs(transform_code_dir)
    
for split_code_name in split_code_name:
    if not os.path.exists(os.path.join(transform_code_dir, split_code_name)):
        os.makedirs(os.path.join(transform_code_dir, split_code_name))

DO_TRAIN = True
DO_TRANSFORM = True

# load corpus
def to_pua(line):
    return "".join(chr(int(tok) + 0xE000) for tok in line.split())

def pua_to_str(line):
    return "".join(chr(ord(tok) - 0xE000) for tok in line)


if DO_TRAIN:
# train sentencepiece model
    print("Length of training data for subword modeling: ", len(code_files))

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        tmp_path = temp_file.name
        for idx, code_file in enumerate(code_files):
            with open(code_file, "r") as f:
                data = f.read().replace("\n", " ")
                data = to_pua(data)
            temp_file.write(data + "\n")
        
    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        max_sentence_length=10000,
        # shuffle_input_sentence=True,
    )
    
if DO_TRANSFORM:
    # transform code files
    if not os.path.exists(f"{model_prefix}.model"):
        raise FileNotFoundError(f"{model_prefix}.model not found")
    
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    
    for code_file in tqdm(code_files, desc="Transforming code files"):
        with open(code_file, "r") as f:
            data = f.read().replace("\n", " ")
            data = to_pua(data)
            
            # transform code file
            transformed_code = sp.encode(data, out_type=int)
            
        save_code_path = code_file.replace(code_dir, transform_code_dir)
        # transformed_code = np.array(transformed_code).long()
        np.savetxt(save_code_path, transformed_code, fmt="%i")





