
## DGB code
# config = yaml.safe_load(config_filepath)
# print("Config file of Dynamic values file loaded.")
import json
with open("./dynamic_values.json", 'r') as file:
    dynamic_values = json.load(file)
print("the dynamic values are",dynamic_values)
# Executing the models based on the given input
# !pip install openpyxl #hiclass 
import hiclass
import sys
sys.path.append("./helper")
sys.path.append("./modules")
from modules.run_workflows import (
    run_logreg_workflow_from_widgets,
    run_rf_workflow_from_widgets,
    run_sgd_workflow_from_widgets 
)

# Dictionary to collect trained models
trained_models = {}

# Iterate over selected classifiers from widgets
for classifier_name in dynamic_values['base_classifiers']:
    clf = classifier_name.lower()

    if clf == 'logisticregression':
        print("⚙️ Running Logistic Regression workflow...")
        model = run_logreg_workflow_from_widgets(dynamic_values)
        trained_models['LogisticRegression'] = model

    elif clf == 'randomforest':
        print("🌲 Running Random Forest workflow...")
        model = run_rf_workflow_from_widgets(dynamic_values)
        trained_models['RandomForest'] = model

    elif clf == 'sgd':
        print("⚙️ Running SGD workflow...")
        model = run_sgd_workflow_from_widgets(dynamic_values)
        trained_models['SGD'] = model

    else:
        print(f"❓ Unknown classifier selected: {classifier_name}")

import pandas as pd

def evaluate_and_compare_models(trained_models, X_test_text, y_test, output_path="outputs/model_comparison.xlsx"):
    """
    Evaluates all trained models, compares their metrics, and saves results to Excel.

    Parameters:
    - trained_models: dict of {model_name: model_instance}
    - X_test_text: text input for prediction
    - y_test: true hierarchical labels
    - output_path: where to save the comparison Excel file

    Returns:
    - summary_df: DataFrame of all model metrics
    """
    all_metrics = []

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for model_name, model in trained_models.items():
            print(f"🔍 Evaluating {model_name}...")
            model.predict(X_test_text)
            metrics = model.evaluate(y_test)

            flat_metrics = {
                'model': model_name,
                'hierarchical_f1': metrics['hierarchical_f1'],
                'hierarchical_precision': metrics['hierarchical_precision'],
                'hierarchical_recall': metrics['hierarchical_recall'],
                'exact_path_accuracy': metrics['exact_path_accuracy']
            }

            # Include level-wise accuracy
            for level, acc in metrics['level_accuracy'].items():
                flat_metrics[f'accuracy_{level}'] = acc

            all_metrics.append(flat_metrics)

            # Save full metric details as a sheet
            pd.DataFrame([flat_metrics]).to_excel(writer, sheet_name=model_name, index=False)

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_excel(output_path, sheet_name="Summary", index=False)

    print(f"✅ Evaluation comparison saved to: {output_path}")
    return summary_df

# This assumes all models were trained and test data is ready
from helper import prepare_text_and_labels, build_dataset_sources_and_labels, build_dataset_map  #helper.helper
from IPython.display import display
# Step 1: Get test data
source_dict, label_dict = build_dataset_sources_and_labels(dynamic_values)
dataset_map = build_dataset_map(source_dict, label_dict)
_, X_test_text, _, y_test = prepare_text_and_labels(dynamic_values, dataset_map)

# Step 2: Evaluate all
results_df = evaluate_and_compare_models(trained_models, X_test_text, y_test)
display(results_df)
