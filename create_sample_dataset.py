import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

def generate_sample_dataset():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    # Map target 0 -> Malignant, 1 -> Benign
    df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
    df = df.drop(columns=["target"])
    
    datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    output_path = os.path.join(datasets_dir, "tcga_brca_sample.csv")
    
    df.to_csv(output_path, index=False)
    print(f"Generated sample dataset at {output_path} with {df.shape[0]} rows and {df.shape[1]} columns.")

if __name__ == "__main__":
    generate_sample_dataset()
