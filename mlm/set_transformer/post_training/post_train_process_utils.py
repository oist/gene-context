import os
import sys
import requests
import pandas as pd

import torch
from torch.utils.data import  DataLoader

parent_directory = os.path.abspath(os.path.join('..'))
grandparent_directory = os.path.abspath(os.path.join(parent_directory, '..'))

sys.path.append(parent_directory)
sys.path.append(grandparent_directory)

from set_transformer.utils.architecture import GenomeSetTransformer
from data_processing_utils.data_processing_functions import GenomeDataset, collate_genomes

def generate_noisy_dataset(df, global_vocab, batch_size, pad_idx, fn_rate, fp_rate, count_noise_std=0, random_state=42):
     dataset = GenomeDataset(df, global_vocab,
                               false_negative_rate=fn_rate, false_positive_rate=fp_rate,
                               count_noise_std=count_noise_std, random_state=random_state)
     dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_genomes(batch, pad_idx=pad_idx))    
     return dataset_loader

def load_model(path, global_vocab, device):
# Load pretrained SetTransformer
    model = GenomeSetTransformer(vocab_size=len(global_vocab), d_model=124,
                                                num_heads=4, num_sab=2, dropout=0.1)

    # Load SetTransformer weights using map_location=device
    model_state = torch.load(path, map_location=device,weights_only=True)   #COG_high3_256_4_8_BCE_40.pth
    # Remove the "module." prefix if trained with DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in model_state.items():
        new_key = key.replace("module.", "")  # Remove "module." prefix
        new_state_dict[new_key] = value

    # Load into model
    model.load_state_dict(new_state_dict)
    model.to(device)
    return model


def create_cogs_with_functions_df():

    # URL for the COG definitions file
    url = "https://ftp.ncbi.nih.gov/pub/COG/COG2020/data/cog-20.def.tab"
    response = requests.get(url)
    if response.status_code == 200:
        with open("cog-20.def.tab", "wb") as f:
            f.write(response.content)
        print("Downloaded cog-20.def.tab successfully.")
    else:
        raise Exception(f"Error downloading file: HTTP {response.status_code}")
    
    # Read the tab-delimited file (it has no header)
    df_cog = pd.read_csv("cog-20.def.tab", sep="\t", header=None, engine="python", encoding="latin1")
    # Assign column names; based on the provided sample:
    df_cog = pd.read_csv("cog-20.def.tab", sep="\t", header=None, engine="python", encoding="latin1")
    df_cog.columns = ["COG_ID", "Category", "Description", "Gene_Symbol", "Function", "Extra", "PDB_ID"]
    # ---------------------------
    # 2. Map to Canonical Metacategories
    # ---------------------------
    # Define canonical mapping for each letter.
    meta_map = {
        "S": "Poorly Characterized", "R": "Poorly Characterized",
        "Q": "Metabolism", "P": "Metabolism", "I": "Metabolism",
        "H": "Metabolism", "F": "Metabolism", "E": "Metabolism",
        "G": "Metabolism", "C": "Metabolism",
        "O": "Cellular Processes & Signaling", "U": "Cellular Processes & Signaling", "W": "Cellular Processes & Signaling",
        "N": "Cellular Processes & Signaling", "M": "Cellular Processes & Signaling", "T": "Cellular Processes & Signaling",
        "V": "Cellular Processes & Signaling", "D": "Cellular Processes & Signaling",
        "B": "Information Storage & Processing", "L": "Information Storage & Processing", "K": "Information Storage & Processing",
        "A": "Information Storage & Processing", "J": "Information Storage & Processing"
    }

    def assign_metacategory(cat_str, meta_map):
        """
        For a given category string (which may contain multiple letters),
        assign a canonical metacategory using majority rule.
        """
        counts = {}
        for letter in str(cat_str):
            if letter in meta_map:
                mc = meta_map[letter]
                counts[mc] = counts.get(mc, 0) + 1
        if counts:
            return max(counts, key=counts.get)
        else:
            return None

    # Process all rows: keep original Category and compute Meta_Category.
    functional_df = df_cog[["COG_ID", "Category"]].copy()
    functional_df["Meta_Category"] = functional_df["Category"].apply(lambda x: assign_metacategory(x, meta_map))
    functional_df = functional_df.dropna(subset=["Meta_Category"])
    functional_df = functional_df.sort_values("COG_ID").reset_index(drop=True)
    functional_df["COG_Index"] = functional_df.index
    print("First few rows of functional_df with Meta_Category:")
    return functional_df


