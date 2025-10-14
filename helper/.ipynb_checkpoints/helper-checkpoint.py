# -----------------------------------------------------------------------------
# helper.py
#
# Author: Najmeh Samadiani <najmeh.samadiani@abs.gov.au>
# Created: 2025
# Description: Dataset preparation, model pipeline utilities, and requirement management
# -----------------------------------------------------------------------------

"""
helper.py

This module contains utility functions to support HiClass-based hierarchical
classification workflows. It includes tools for dataset loading, label parsing,
pipeline building, and environment setup.

Included Functions:
-------------------

1. `build_dataset_sources_and_labels(dynamic_values)`:
   - Dynamically builds a mapping of dataset names to their respective file paths
   - Extracts user-defined label columns for each dataset via widget input
   - Useful when switching between datasets like 'dataset1' and 'dataset2' in the UI

2. `build_dataset_map(source_dict, label_dict)`:
   - Loads each dataset (CSV) using pandas
   - Extracts only the specified label columns for use in training
   - Returns a mapping of dataset name → (train_df, test_df) tuple

3. `prepare_text_and_labels(dynamic_values, dataset_map)`:
   - Uses the selected dataset and user choice of `text_column` ('Title', 'Text', or 'Both')
   - Returns X_train_text, X_test_text, y_train, and y_test ready for model training

4. `build_safe_pipeline(...)`:
   - Wraps HiClass classifier setup
   - If the selected probability combiner (e.g., 'geometric') causes divide-by-zero issues,
     it falls back to a safer alternative ('arithmetic')
   - Ensures the training pipeline completes without interrupting the notebook

5. `install_requirements(requirements_file='requirements.txt')`:
   - Programmatically installs Python packages listed in a requirements file
   - Useful for runtime setup in cloud notebooks like SageMaker or Colab

Typical Usage:
---------------
source_dict, label_dict = build_dataset_sources_and_labels(dynamic_values)
dataset_map = build_dataset_map(source_dict, label_dict)
X_train_text, X_test_text, y_train, y_test = prepare_text_and_labels(dynamic_values, dataset_map)

This module supports modular, scalable setup for multi-dataset experiments and hierarchical pipelines.
"""

import pandas as pd
import subprocess
import sys
import os
from sklearn.model_selection import train_test_split

# --- Dataset builder and data preparation helpers ---
def build_dataset_sources_and_labels(dynamic_values):
    """
    Load datasets from file and return source_dict and label_dict
    for use in build_dataset_map(). Supports 'dataset1' for now.
    """
    source_dict = {}
    label_dict = {}

    if 'dataset1' in dynamic_values['dataset']:
        train_data_location = './data/kaggle_data/train.csv'
        test_data_location = './data/kaggle_data/test.csv'
        train_df = pd.read_csv(train_data_location)
        test_df = pd.read_csv(test_data_location)
        source_dict['dataset1'] = (train_df, test_df)

    if 'dataset2' in dynamic_values['dataset']:
        coicop_path = './data/coicop_data/coicop_5d_condensed.txt'
        texts, labels = [], []
        with open(coicop_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    code_full, text = line.strip().split(":", 1)
                    code = code_full.strip().lstrip("CP")
                    text = text.strip()
                    if len(code) == 5:
                        dot_code = '.'.join([code[:2], code[2], code[3], code[4]])
                        parts = dot_code.split(".")
                        label_path = ['.'.join(parts[:i+1]) for i in range(len(parts))]
                        texts.append(text)
                        labels.append(label_path)

        # Create DataFrame and split
        df = pd.DataFrame({"Title": texts, "coicop_label": labels})
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        source_dict['dataset2'] = (train_df, test_df)

    # Parse label text inputs into lists
    for ds_key in source_dict:
        label_key = f'label_columns_{ds_key}'
        label_dict[ds_key] = dynamic_values.get(label_key, [])

    return source_dict, label_dict


def build_dataset_map(source_dict, label_dict):
    """
    Returns a dataset map for the selected dataset
    """
    dataset_map = {}
    for name in source_dict:
        train_df, test_df = source_dict[name]
        labels = label_dict.get(name, [])
        dataset_map[name] = {
            'train': train_df,
            'test': test_df,
            'labels': labels
        }
    return dataset_map


# def prepare_text_and_labels(dynamic_values, dataset_map):
#     """
#     Returns X_train_text, X_test_text, y_train, y_test for the selected dataset.
#     Assumes dataset_map is a dict like:
#         {'dataset1': {'train': ..., 'test': ..., 'labels': [...]}, ...}
#     """
#     selected = dynamic_values['dataset']
#     ds = dataset_map[selected]

#     train_df = ds['train']
#     test_df = ds['test']
#     label_cols = ds['labels']
#     text_col = dynamic_values['text_column']

#     if text_col == 'Both':
#         X_train_text = train_df['Title'].fillna('') + ' ' + train_df['Text'].fillna('')
#         X_test_text = test_df['Title'].fillna('') + ' ' + test_df['Text'].fillna('')
#     else:
#         X_train_text = train_df[text_col].fillna('')
#         X_test_text = test_df[text_col].fillna('')

#     y_train = train_df[label_cols]
#     y_test = test_df[label_cols]

#     return X_train_text, X_test_text, y_train, y_test

def prepare_text_and_labels(dynamic_values, dataset_map):
    """
    Returns X_train_text, X_test_text, y_train, y_test for the selected dataset.
    Supports multiple datasets with flexible column handling:
    - dataset1: expects label columns and separate text fields
    - dataset2: uses 'text' and 'hierarchical_label' columns directly
    """
    selected = dynamic_values['dataset']
    ds = dataset_map[selected]

    train_df = ds['train']
    test_df = ds['test']
    text_col = dynamic_values.get('text_column', 'text')  # fallback for dataset2

    if selected == 'dataset1':
        label_cols = ds['labels']
        if text_col == 'Both':
            X_train_text = train_df['Title'].fillna('') + ' ' + train_df['Text'].fillna('')
            X_test_text = test_df['Title'].fillna('') + ' ' + test_df['Text'].fillna('')
        else:
            X_train_text = train_df[text_col].fillna('')
            X_test_text = test_df[text_col].fillna('')
        y_train = train_df[label_cols]
        y_test = test_df[label_cols]

    elif selected == 'dataset2':
        # Direct access for COICOP-style dataset
        X_train_text = train_df['Title'].fillna('')
        X_test_text = test_df['Title'].fillna('')
        
        max_depth = max(len(path) for path in train_df['coicop_label'])
        col_names = [f"Level_{i+1}" for i in range(max_depth)]
        
        y_train = pd.DataFrame(train_df['coicop_label'].tolist(), columns=col_names)
        y_test = pd.DataFrame(test_df['coicop_label'].tolist(), columns=col_names)

    else:
        raise ValueError(f"Unsupported dataset key: {selected}")

    return X_train_text, X_test_text, y_train, y_test


def install_requirements(requirements_file='requirements.txt'):
    """
    Installs all packages from requirements.txt with minimal output
    """
    try:
        if not os.path.exists(requirements_file):
            print("❌ requirements.txt file not found!")
            return
        
        print("🚀 Installing packages from requirements.txt...")
        
        # Install all requirements at once
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', requirements_file
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All packages installed successfully!")
        else:
            print("❌ Some packages failed to install:")
            print(result.stderr)
            
        # Verify installation (silent)
        check_result = subprocess.run([sys.executable, '-m', 'pip', 'check'], 
                                    capture_output=True, text=True)
        
        if check_result.returncode == 0:
            print("✅ All dependencies are compatible!")
        else:
            print("⚠️  Dependency conflicts detected:")
            print(check_result.stdout)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def build_safe_pipeline(
    X_train_text,
    y_train,
    vectorizer,
    strategy_class,
    base_classifier,
    calibration_method=None,
    probability_combiner='geometric'
):
    from sklearn.pipeline import Pipeline
    import numpy as np

    def make_pipeline(prob_comb):
        hiclassifier = strategy_class(
            local_classifier=base_classifier,
            calibration_method=calibration_method,
            probability_combiner=prob_comb
        )
        return Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', hiclassifier)
        ])

    # RandomForestClassifier.predict_proba() often returns hard zeros and 
    # the 'geometric' method (GeometricMeanCombiner) tries to np.log(0) which is undefined (-inf)
    def check_zero_probs(pipeline, X_text_sample, n_check=20):
        """
        Checks if any predicted probabilities are exactly zero, which may
        cause issues with the 'geometric' probability combiner (e.g., log(0)).
    
        Parameters:
        - pipeline: fitted pipeline with vectorizer and classifier
        - X_text_sample: input text to test probability output
        - n_check: number of samples to test (default: 20)
    
        Returns:
        - True if any 0 probabilities are detected, else False
        """
        try:
            proba = pipeline.named_steps['classifier'].predict_proba(
                pipeline.named_steps['vectorizer'].transform(X_text_sample[:n_check])
            )
            # Flatten and check for exact 0s
            flat_probs = np.concatenate([np.array(p).flatten() for p in proba])
            return (flat_probs == 0).any()
        except Exception:
            return True  # if we can’t compute proba, better to fallback

    # Try geometric first
    current_combiner = probability_combiner
    pipeline = make_pipeline(current_combiner)
    pipeline.fit(X_train_text, y_train)

    if current_combiner == 'geometric' and check_zero_probs(pipeline, X_train_text):
        print("⚠️ Detected 0 probabilities with 'geometric' combiner. Switching to 'arithmetic'.")
        current_combiner = 'arithmetic'
        pipeline = make_pipeline(current_combiner)
        pipeline.fit(X_train_text, y_train)

    return pipeline, current_combiner
