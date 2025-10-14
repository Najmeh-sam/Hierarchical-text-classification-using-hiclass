# -----------------------------------------------------------------------------
# hiclass_rf_model.py
#
# Author: Najmeh Samadiani <najmeh.samadiani@abs.gov.au>
# Created: 2025
# Description: Hierarchical classification using RandomForest and HiClass
# -----------------------------------------------------------------------------
"""
This module defines the HiClassRFModel class for performing hierarchical
classification using a RandomForestClassifier as the base classifier within
a HiClass strategy.

Features:
- Supports configurable HiClass strategies: 'lcppn', 'lcpn', 'lcpl'
- Applies TF-IDF vectorization for text preprocessing
- Supports probability-based prediction with safe fallback from geometric to arithmetic combination
- Accepts Random Forest hyperparameters including:
    - n_estimators: number of trees
    - max_depth: maximum depth of each tree
    - criterion: 'gini' or 'entropy' for split quality
- Provides methods for training, prediction, evaluation, saving, loading, and visualizing results
- Includes hierarchical evaluation metrics and confusion matrix export

Typical usage:
    model = HiClassRFModel(...)
    model.train(X_train_text, y_train)
    model.predict(X_test_text)
    metrics = model.evaluate(y_test)
    model.save("path/to/model.joblib")

This model is suitable for multi-level classification problems where labels are structured hierarchically.
"""

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from hiclass import (
    LocalClassifierPerParentNode,
    LocalClassifierPerNode,
    LocalClassifierPerLevel
)
from helper import build_safe_pipeline
from hiclass.metrics import f1, precision, recall
import pandas as pd
import joblib
import os


class HiClassRFModel:
    def __init__(
        self,
        strategy='lcppn',
        rf_params=None,
        calibration_method=None,
        probability_combiner=None,
        tfidf_params=None
    ):
        strategy_map = {
            'lcppn': LocalClassifierPerParentNode,
            'lcpn': LocalClassifierPerNode,
            'lcpl': LocalClassifierPerLevel
        }
        self.strategy_class = strategy_map.get(strategy, LocalClassifierPerParentNode)
        self.calibration_method = calibration_method
        self.probability_combiner = probability_combiner
        self.criterion = rf_params.get('criterion', 'gini')
        self.pipeline = None
        self.predictions = None
        self.confidence_df = None

        self.tfidf_params = tfidf_params or {
            'stop-words': 'english',
            'ngram-min': 1,
            'ngram-max': 2,
            'max-df': 0.5,
            'min-df': 1,
            'max-features': 50000
        }

        self.rf_params = rf_params or {
            'n_estimators': 100,
            'max_depth': None,
            'criterion': 'gini',
            'random_state': 42
        }

    def train(self, X_train_text, y_train):
        vectorizer = TfidfVectorizer(
            stop_words=self.tfidf_params['stop-words'],
            ngram_range=(self.tfidf_params['ngram-min'], self.tfidf_params['ngram-max']),
            max_df=self.tfidf_params['max-df'],
            min_df=self.tfidf_params['min-df'],
            max_features=self.tfidf_params['max-features']
        )

        #base_clf = RandomForestClassifier(**self.rf_params)
        base_clf = RandomForestClassifier(
            n_estimators=self.rf_params.get('n_estimators', 100),
            max_depth=self.rf_params.get('max_depth', None),
            criterion=self.criterion,
            random_state=self.rf_params.get('random_state', 42)
        )

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
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet.")
        self.predictions = self.pipeline.predict(X_test_text)
        return self.predictions

    def evaluate(self, y_test):
        if self.predictions is None:
            raise ValueError("Run predict() first.")
        predictions_df = pd.DataFrame(self.predictions, columns=y_test.columns)

        metrics = {
            'level_accuracy': {
                level: accuracy_score(y_test[level], predictions_df[level])
                for level in y_test.columns
            },
            'hierarchical_f1': f1(y_test, predictions_df),
            'hierarchical_precision': precision(y_test, predictions_df),
            'hierarchical_recall': recall(y_test, predictions_df),
            'exact_path_accuracy': (predictions_df == y_test).all(axis=1).mean()
        }

        self.metrics_ = metrics
        return metrics

    def analyze_confidence_scores(self, X_test_text, y_test, top_n=5):
        if self.pipeline is None:
            raise ValueError("Model not trained.")
        if self.predictions is None:
            raise ValueError("Run predict() before confidence analysis.")
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

        self.confidence_df = confidence_df

        print(f"🔍 Avg top confidence: {confidence_df['Confidence'].mean():.4f}")
        top_conf = confidence_df[confidence_df['Correct']].sort_values(by='Confidence', ascending=False).head(top_n)
        print(f"🎯 Top {top_n} confident correct samples:\n{top_conf}")

        return confidence_df

    def get_confusion_matrices(self, y_test):
        if self.predictions is None:
            raise ValueError("Run predict() first.")
        predictions_df = pd.DataFrame(self.predictions, columns=y_test.columns)

        return {
            level: confusion_matrix(y_test[level], predictions_df[level])
            for level in y_test.columns
        }

    def print_confusion_matrices(self, y_test, label_names_dict=None, export_path=None):
        cm_dict = self.get_confusion_matrices(y_test)
        predictions_df = pd.DataFrame(self.predictions, columns=y_test.columns)

        writer = pd.ExcelWriter(export_path, engine='openpyxl') if export_path else None

        print("🧩 Confusion Matrices:\n")
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
            print(f"📁 Confusion matrices saved to: {export_path}")

    def save(self, filepath):
        if self.pipeline is None:
            raise ValueError("No trained model to save.")

        joblib.dump({
            'pipeline': self.pipeline,
            'tfidf_params': self.tfidf_params,
            'rf_params': self.rf_params,
            'strategy': self.strategy_class.__name__,
            'calibration_method': self.calibration_method,
            'probability_combiner': self.probability_combiner,
            'predictions': self.predictions,
            'evaluation_metrics': getattr(self, 'metrics_', None),
            'confidence_df': self.confidence_df
        }, filepath)
        print(f"✅ Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at: {filepath}")

        saved = joblib.load(filepath)

        instance = cls(
            strategy=saved.get('strategy', 'lcppn').lower(),
            rf_params=saved.get('rf_params'),
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
