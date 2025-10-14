# -----------------------------------------------------------------------------
# run_workflows.py
#
# Author: Najmeh Samadiani <najmeh.samadiani@abs.gov.au>
# Created: 2025
# Description: Model training, evaluation, and saving workflows for HiClass models
# -----------------------------------------------------------------------------
"""
run_workflows.py

This module provides high-level workflow functions to train, evaluate, and
store hierarchical classification models using different base classifiers
within the HiClass framework.

Each function reads interactive configuration values from the dynamic input
widgets (`dynamic_values`), builds and trains a model using the appropriate
HiClass variant, and optionally saves outputs such as the trained model,
evaluation metrics, and confusion matrices.

Included Workflows:
- `run_logreg_workflow_from_widgets()`: trains a HiClass model with LogisticRegression
- `run_rf_workflow_from_widgets()`: trains a HiClass model with RandomForestClassifier
- `run_sgd_workflow_from_widgets()`: trains a HiClass model with SGDClassifier, optionally with calibration

Features:
- Reads user-selected TF-IDF, classifier, and HiClass strategy settings
- Automatically applies safe training logic for probability combiners
- Evaluates hierarchical metrics: precision, recall, F1, level accuracy, and exact path accuracy
- Optionally saves models and confusion matrices with timestamped filenames
- Central entry point for managing multiple workflows through the widget interface

Typical Usage:
---------------
# From within a notebook or script:
from run_workflows import run_logreg_workflow_from_widgets
model = run_logreg_workflow_from_widgets(dynamic_values)

# Or call all selected workflows:
for clf in dynamic_values['base_classifiers']:
    if clf.lower() == 'sgd':
        model = run_sgd_workflow_from_widgets(dynamic_values)

This module enables unified, reproducible training pipelines across different classifier backends under a consistent HiClass interface.
"""

from helper import (
        build_dataset_sources_and_labels,
        build_dataset_map,
        prepare_text_and_labels
    )
from hiclass_logreg_model import HiClassLogRegModel
from hiclass_rf_model import HiClassRFModel
from hiclass_sgd_model import HiClassSGDModel

import datetime
import os

def run_logreg_workflow_from_widgets(dynamic_values):
    """
    Executes the full workflow for training, evaluating, and reporting
    a HiClass hierarchical model using Logistic Regression as the base classifier.

    This function:
    - Reads user input from the interactive widget configuration (dynamic_values)
    - Loads and prepares the selected dataset
    - Applies TF-IDF text preprocessing using user-defined parameters
    - Trains the HiClass + LogisticRegression model
    - Evaluates the model using hierarchical metrics
    - Displays confidence analysis and confusion matrices
    - Exports confusion matrices to an Excel file

    Parameters:
    - dynamic_values (dict): Dictionary of widget values returned from build_model_configuration_widgets()

    Returns:
    - model (HiClassLogRegModel): The trained model instance for further use (e.g., saving, re-evaluation)
    """

    # 1. Load and prepare data - all functions have been defined in the 'helper.py' module
    source_dict, label_dict = build_dataset_sources_and_labels(dynamic_values)
    dataset_map = build_dataset_map(source_dict, label_dict)
    X_train_text, X_test_text, y_train, y_test = prepare_text_and_labels(dynamic_values, dataset_map)

    # 2. Extract TF-IDF and model parameters
    tfidf_params = {
        'stop-words': dynamic_values['stop_words'],
        'ngram-min': dynamic_values['ngram_min'],
        'ngram-max': dynamic_values['ngram_max'],
        'max-df': dynamic_values['max_df'],
        'min-df': dynamic_values['min_df'],
        'max-features': dynamic_values['max_features']
    }

    calibration_method = (
        dynamic_values['calibration_method']
        if dynamic_values['calib_prob_control'] in ['calibration only', 'both']
        else None
    )
    probability_combiner = (
        dynamic_values['probability_combiner']
        if dynamic_values['calib_prob_control'] in ['probability only', 'both']
        else None
    )

    # 3. Train and test the model
    model = HiClassLogRegModel(
        strategy=dynamic_values['Hiclass_strategy'],
        max_iter=dynamic_values['SGD_max_iter'],
        calibration_method=calibration_method,
        probability_combiner=probability_combiner,
        tfidf_params=tfidf_params
    )

    model.train(X_train_text, y_train)
    model.predict(X_test_text)

    # 4. Evaluate and show results
    metrics = model.evaluate(y_test)
    print(f"📊 Evaluation Results for the Logistic Regression with Calibration ({calibration_method}) and Probability ({probability_combiner}:) ")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 5. Confidence scores (if probability is enabled)
    if dynamic_values['calib_prob_control'] in ['probability only', 'both']:
        model.analyze_confidence_scores(X_test_text, y_test)

    # Save the model in the path
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"rf_model_{timestamp}.joblib")
    cm_path = os.path.join(output_dir, f"rf_confusion_{timestamp}.xlsx")
    model.save(model_path)
    model.print_confusion_matrices(y_test, export_path=cm_path)

    return model


def run_rf_workflow_from_widgets(dynamic_values):
    """
    Executes the full workflow for training, evaluating, and reporting
    a HiClass hierarchical model using Random Forest as the base classifier.

    This function:
    - Reads user input from the interactive widget configuration (dynamic_values)
    - Loads and prepares the selected dataset
    - Applies TF-IDF text preprocessing using user-defined parameters
    - Trains the HiClass + RandomForest model
    - Evaluates the model using hierarchical metrics
    - Displays confidence analysis and confusion matrices
    - Exports confusion matrices to an Excel file

    Parameters:
    - dynamic_values (dict): Dictionary of widget values returned from build_model_configuration_widgets()

    Returns:
    - model (HiClassRFModel): The trained model instance for further use (e.g., saving, re-evaluation)
    """

    # 1. Load and prepare data
    source_dict, label_dict = build_dataset_sources_and_labels(dynamic_values)
    dataset_map = build_dataset_map(source_dict, label_dict)
    X_train_text, X_test_text, y_train, y_test = prepare_text_and_labels(dynamic_values, dataset_map)

    # 2. Extract TF-IDF and model parameters
    tfidf_params = {
        'stop-words': dynamic_values['stop_words'],
        'ngram-min': dynamic_values['ngram_min'],
        'ngram-max': dynamic_values['ngram_max'],
        'max-df': dynamic_values['max_df'],
        'min-df': dynamic_values['min_df'],
        'max-features': dynamic_values['max_features']
    }

    # Interpret 0 as letting the model decide (i.e., max_depth=None)
    rf_max_depth = dynamic_values['RF_max_depth']
    rf_max_depth = None if rf_max_depth == 0 else rf_max_depth
    
    rf_params = {
        'n_estimators': dynamic_values['RF_estimators'],
        'max_depth': rf_max_depth,
        'criterion': dynamic_values['RF_criterion'],
        'random_state': dynamic_values['Random_state']
    }

    calibration_method = (
        dynamic_values['calibration_method']
        if dynamic_values['calib_prob_control'] in ['calibration only', 'both']
        else None
    )
    probability_combiner = (
        dynamic_values['probability_combiner']
        if dynamic_values['calib_prob_control'] in ['probability only', 'both']
        else None
    )

    # 3. Train the model
    model = HiClassRFModel(
        strategy=dynamic_values['Hiclass_strategy'],
        rf_params=rf_params,
        calibration_method=calibration_method,
        probability_combiner=probability_combiner,
        tfidf_params=tfidf_params
    )

    model.train(X_train_text, y_train)
    model.predict(X_test_text)

    # 4. Evaluate and show results
    metrics = model.evaluate(y_test)
    print(f"📊 Evaluation Results for the Random Forest with Calibration ({calibration_method}) and Probability ({probability_combiner}):")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 5. Confidence scores (if probability is enabled)
    if dynamic_values['calib_prob_control'] in ['probability only', 'both']:
        model.analyze_confidence_scores(X_test_text, y_test)

    # 6. Save the model
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"logistic_reg_model_{timestamp}.joblib")
    cm_path = os.path.join(output_dir, f"logistic_reg_confusion_{timestamp}.xlsx")
    model.save(model_path)
    model.print_confusion_matrices(y_test, export_path=cm_path)

    return model

def run_sgd_workflow_from_widgets(dynamic_values):
    """
    Executes the full workflow for training, evaluating, and reporting
    a HiClass hierarchical model using SGDClassifier as the base classifier.

    - Uses hinge or log loss (probability only if log is selected)
    - Optionally applies calibration and global bootstrapping

    Returns:
    - model (HiClassSGDModel): trained model instance
    """
    # 1. Load and prepare data
    source_dict, label_dict = build_dataset_sources_and_labels(dynamic_values)
    dataset_map = build_dataset_map(source_dict, label_dict)
    X_train_text, X_test_text, y_train, y_test = prepare_text_and_labels(dynamic_values, dataset_map)

    # 2. TF-IDF parameters
    tfidf_params = {
        'stop-words': dynamic_values['stop_words'],
        'ngram-min': dynamic_values['ngram_min'],
        'ngram-max': dynamic_values['ngram_max'],
        'max-df': dynamic_values['max_df'],
        'min-df': dynamic_values['min_df'],
        'max-features': dynamic_values['max_features']
    }

    # 3. Determine calibration/probability
    loss = dynamic_values['SGD_loss']
    calibration_method = None
    probability_combiner = None
    if loss != 'hinge':
        if dynamic_values['calib_prob_control'] in ['calibration only', 'both']:
            calibration_method = dynamic_values['calibration_method']
        if dynamic_values['calib_prob_control'] in ['probability only', 'both']:
            probability_combiner = dynamic_values['probability_combiner']

    # 4. Define model
    model = HiClassSGDModel(
        strategy=dynamic_values['Hiclass_strategy'],
        svm_loss=loss,
        svm_class_weight=dynamic_values['SGD_class_weight'],
        max_iter=dynamic_values['SGD_max_iter'],
        calibration_method=calibration_method,
        probability_combiner=probability_combiner,
        tfidf_params=tfidf_params,
        random_state=dynamic_values['Random_state']
    )

    model.train(X_train_text, y_train)
    model.predict(X_test_text)

    # 5. Evaluate and show results
    metrics = model.evaluate(y_test)
    print(f"\n📊 Evaluation Results for SGD ({loss}) with Calibration ({calibration_method}) and Probability ({probability_combiner}):")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 6. Confidence scores
    if probability_combiner:
        model.analyze_confidence_scores(X_test_text, y_test)

    # 7. Save model and confusion matrix

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"sgd_model_{timestamp}.joblib")
    cm_path = os.path.join(output_dir, f"sgd_confusion_{timestamp}.xlsx")
    model.save(model_path)
    model.print_confusion_matrices(y_test, export_path=cm_path)

    return model
