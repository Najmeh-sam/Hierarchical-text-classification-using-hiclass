# -----------------------------------------------------------------------------
# hiclass_logreg_model.py
#
# Author: Najmeh Samadiani <najmeh.samadiani@abs.gov.au>
# Created: 2025
# Description: HiClass model using LogisticRegression as the base classifier
# -----------------------------------------------------------------------------

"""
hiclass_logreg_model.py

This module defines the HiClassLogRegModel class, which builds and evaluates
a hierarchical text classification pipeline using scikit-learn's
LogisticRegression as the base classifier in conjunction with the HiClass
hierarchical classification framework.

Features:
- Supports HiClass strategies: 'lcppn', 'lcpn', 'lcpl'
- Integrates TF-IDF vectorization for processing textual input
- Optionally supports probability calibration and hierarchical probability combiners
- Provides model training, prediction, evaluation, confidence analysis, and
  export of hierarchical confusion matrices
- Includes methods to save/load the pipeline and associated metadata

Typical usage:
    model = HiClassLogRegModel(...)
    model.train(X_train_text, y_train)
    model.predict(X_test_text)
    metrics = model.evaluate(y_test)
    model.save("path/to/model.joblib")

This model is suitable for structured classification tasks where categories
exist at multiple levels of a hierarchy and where interpretable, calibrated
outputs are desirable.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from hiclass import (
    LocalClassifierPerParentNode,
    LocalClassifierPerNode,
    LocalClassifierPerLevel
)
from helper import build_safe_pipeline

from sklearn.metrics import accuracy_score
from hiclass.metrics import f1, precision, recall
from sklearn.metrics import confusion_matrix
import pandas as pd
import joblib
from openpyxl import Workbook
import os

class HiClassLogRegModel:
    def __init__(
        self,
        strategy='lcppn',
        max_iter=1000,
        calibration_method=None,
        probability_combiner=None,
        tfidf_params=None
    ):
        """
        Initialize model with configurable HiClass strategy and TF-IDF parameters.
        """
        strategy_map = {
            'lcppn': LocalClassifierPerParentNode,
            'lcpn': LocalClassifierPerNode,
            'lcpl': LocalClassifierPerLevel
        }

        self.strategy_class = strategy_map.get(strategy, LocalClassifierPerParentNode)
        self.max_iter = max_iter
        self.calibration_method = calibration_method
        self.probability_combiner = probability_combiner
        self.pipeline = None
        self.predictions = None

        self.tfidf_params = tfidf_params or {
            'stop-words': 'english',
            'ngram-min': 1,
            'ngram-max': 2,
            'max-df': 0.5,
            'min-df': 1,
            'max-features': 50000
        }

    def train(self, X_train_text, y_train):
        """
        Train the pipeline: TF-IDF + HiClass
        Automatically switches geometric combiner to arithmetic if log(0) issues occur.
        """
        import warnings
    
        vectorizer = TfidfVectorizer(
            stop_words=self.tfidf_params['stop-words'],
            ngram_range=(self.tfidf_params['ngram-min'], self.tfidf_params['ngram-max']),
            max_df=self.tfidf_params['max-df'],
            min_df=self.tfidf_params['min-df'],
            max_features=self.tfidf_params['max-features']
        )
    
        base_clf = LogisticRegression(max_iter=self.max_iter)

        self.pipeline, self.probability_combiner = build_safe_pipeline(
            X_train_text=X_train_text,
            y_train=y_train,
            vectorizer=vectorizer,
            strategy_class=self.strategy_class,
            base_classifier=base_clf,
            calibration_method=self.calibration_method,
            probability_combiner=self.probability_combiner
        )

    def predict(self, X_test_text):
        """
        Generate hierarchical predictions.
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet.")
        self.predictions = self.pipeline.predict(X_test_text)
        return self.predictions

    def evaluate(self, y_test):
        """
        Evaluate model using hierarchical metrics.
        """
        if self.predictions is None:
            raise ValueError("No predictions found. Run predict() first.")

        predictions_df = pd.DataFrame(self.predictions, columns=y_test.columns)
        level_accuracies = {
            level: accuracy_score(y_test[level], predictions_df[level])
            for level in y_test.columns
        }

        hierarchical_f1_score = f1(y_test, predictions_df)
        hierarchical_precision_score = precision(y_test, predictions_df)
        hierarchical_recall_score = recall(y_test, predictions_df)
        exact_match_accuracy = (predictions_df == y_test).all(axis=1).mean()

        
        metrics={"level_accuracy": level_accuracies,
            "hierarchical_f1": hierarchical_f1_score,
            "hierarchical_precision": hierarchical_precision_score,
            "hierarchical_recall": hierarchical_recall_score,
            "exact_path_accuracy": exact_match_accuracy
        }

        self.metrics_ = metrics 
        return metrics

    def get_confusion_matrices(self, y_test):
        """
        Compute confusion matrices for each level (e.g., Cat1, Cat2, Cat3).

        Returns:
        - dict of {level: confusion_matrix}
        """
        if self.predictions is None:
            raise ValueError("Run predict() before calling this method.")

        predictions_df = pd.DataFrame(self.predictions, columns=y_test.columns)

        cm_dict = {}
        for level in y_test.columns:
            cm = confusion_matrix(y_test[level], predictions_df[level])
            cm_dict[level] = cm

        return cm_dict
    
    def analyze_confidence_scores(self, X_test_text, y_test, top_n=5):
        """
        Analyze prediction confidence from probability outputs.
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet.")
        if self.predictions is None:
            raise ValueError("Run predict() before analyzing confidence.")
        if not hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            raise ValueError("Classifier does not support predict_proba().")

        probs = self.pipeline.named_steps['classifier'].predict_proba(
            self.pipeline.named_steps['vectorizer'].transform(X_test_text)
        )

        predictions_df = pd.DataFrame(self.predictions, columns=y_test.columns)
        confidence_scores = [max(p) for p in probs]
        correct_predictions = (predictions_df == y_test).all(axis=1)

        confidence_df = pd.DataFrame({
            'Confidence': confidence_scores,
            'Correct': correct_predictions
        })

        self.confidence_df = confidence_df  # Save for export

        print(f"🔍 Average top confidence score: {confidence_df['Confidence'].mean():.4f}")
        top_conf = confidence_df[confidence_df['Correct']].sort_values(by='Confidence', ascending=False).head(top_n)
        print(f"\n🎯 Top {top_n} most confident correct predictions:")
        print(top_conf)

        return confidence_df

    def save(self, filepath):
        """
        Save the trained pipeline, TF-IDF params, predictions, and evaluation metrics.
        """
        if self.pipeline is None:
            raise ValueError("No trained pipeline to save.")

        save_dict = {
            'pipeline': self.pipeline,
            'tfidf_params': self.tfidf_params,
            'strategy': self.strategy_class.__name__,
            'max_iter': self.max_iter,
            'calibration_method': self.calibration_method,
            'probability_combiner': self.probability_combiner,
            'predictions': self.predictions,
            'evaluation_metrics': getattr(self, 'metrics_', None),
            'confidence_df': getattr(self, 'confidence_df', None)
        }

        joblib.dump(save_dict, filepath)
        print(f"✅ Model and metadata saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load a saved HiClassLogRegModel from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at: {filepath}")

        saved = joblib.load(filepath)

        instance = cls(
            strategy=saved.get('strategy', 'lcppn').lower(),
            max_iter=saved.get('max_iter', 1000),
            calibration_method=saved.get('calibration_method'),
            probability_combiner=saved.get('probability_combiner'),
            tfidf_params=saved.get('tfidf_params')
        )

        instance.pipeline = saved['pipeline']
        instance.predictions = saved.get('predictions')
        instance.metrics_ = saved.get('evaluation_metrics')
        instance.confidence_df = saved.get('confidence_df')

        print(f"✅ Model loaded from {filepath}")
        return instance

    def print_confusion_matrices(self, y_test, label_names_dict=None, export_path=None):
        """
        Pretty-print and optionally export confusion matrices per hierarchy level.

        Parameters:
        - y_test: DataFrame of true labels
        - label_names_dict: optional dict of {level: list of label names}
        - export_path: path to save Excel file (e.g., 'confusion_matrices.xlsx')
        """
        cm_dict = self.get_confusion_matrices(y_test)
        predictions_df = pd.DataFrame(self.predictions, columns=y_test.columns)

        writer = pd.ExcelWriter(export_path, engine='openpyxl') if export_path else None

        print("🧩 Confusion Matrices (per level):\n")

        for level in y_test.columns:
            cm = cm_dict[level]
            if label_names_dict and level in label_names_dict:
                labels = label_names_dict[level]
            else:
                all_labels = pd.concat([y_test[level], predictions_df[level]]).unique()
                labels = sorted(all_labels)

            df_cm = pd.DataFrame(cm, index=range(cm.shape[0]), columns=range(cm.shape[1]))
            df_cm.index = labels
            df_cm.columns = labels

            print(f"📘 Level: {level}")
            display(df_cm)
            print("-" * 50)

            if writer:
                df_cm.to_excel(writer, sheet_name=level)

        if writer:
            writer.close()
            print(f"📁 Confusion matrices written to: {export_path}")