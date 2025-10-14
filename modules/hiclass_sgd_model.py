# -----------------------------------------------------------------------------
# hiclass_sgd_model.py
#
# Author: Najmeh Samadiani <najmeh.samadiani@abs.gov.au>
# Created: 2025
# Description: HiClass model using SGDClassifier as the base classifier
# -----------------------------------------------------------------------------
"""
hiclass_sgd_model.py

This module defines the HiClassSGDModel class, which implements a hierarchical
classification pipeline using scikit-learn's SGDClassifier as the base
classifier within the HiClass framework.

Features:
- Supports HiClass strategies: 'lcppn', 'lcpn', 'lcpl'
- Compatible with 'hinge' (SVM-like) and 'log' (logistic) loss functions
- Offers optional calibration using CalibratedClassifierCV for probabilistic output
- Integrates TF-IDF vectorization for textual data
- Provides prediction, evaluation (including hierarchical metrics), and export of confusion matrices
- Includes confidence analysis and model persistence (save/load)

Important Notes:
- Probability-based scoring (e.g., calibration, confidence analysis) is only available when using 'log' loss
- 'hinge' loss disables probability output and calibration automatically

Typical Usage:
---------------
model = HiClassSGDModel(
    strategy='lcppn',
    svm_loss='log',
    calibration_method='isotonic',
    probability_combiner='geometric',
    bootstrap='yes',
    tfidf_params={
        'stop-words': 'english',
        'ngram-min': 1,
        'ngram-max': 2,
        'max-df': 0.5,
        'min-df': 1,
        'max-features': 50000
    }
)

model.train(X_train_text, y_train)
model.predict(X_test_text)
metrics = model.evaluate(y_test)
model.analyze_confidence_scores(X_test_text, y_test)
model.print_confusion_matrices(y_test, export_path='outputs/confusion_sgd.xlsx')
model.save('outputs/sgd_model.joblib')

This model is ideal for scalable, memory-efficient hierarchical classification
tasks with optional probability outputs.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from hiclass import LocalClassifierPerParentNode, LocalClassifierPerNode, LocalClassifierPerLevel
from helper import build_safe_pipeline

class HiClassSGDModel:
    def __init__(
        self,
        strategy='lcppn',
        svm_loss='hinge',
        svm_class_weight='balanced',
        max_iter=1000,
        calibration_method=None,
        probability_combiner=None,
        tfidf_params=None,
        random_state=0
    ):
        """
        Initialize a hierarchical classification model with SGDClassifier.
        """
        strategy_map = {
            'lcppn': LocalClassifierPerParentNode,
            'lcpn': LocalClassifierPerNode,
            'lcpl': LocalClassifierPerLevel
        }
        self.strategy_class = strategy_map.get(strategy, LocalClassifierPerParentNode)
        self.loss = svm_loss
        self.class_weight = svm_class_weight
        self.max_iter = max_iter
        self.random_state = random_state

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
        Train the pipeline: TF-IDF + HiClass + SGDClassifier
        
        - If hinge loss is selected, disables probability & calibration.
        - If calibration is enabled, uses CalibratedClassifierCV.
        - If bootstrap is True, wraps with BaggingClassifier.
        """
        vectorizer = TfidfVectorizer(
            stop_words=self.tfidf_params['stop-words'],
            ngram_range=(self.tfidf_params['ngram-min'], self.tfidf_params['ngram-max']),
            max_df=self.tfidf_params['max-df'],
            min_df=self.tfidf_params['min-df'],
            max_features=self.tfidf_params['max-features']
        )

        base_sgd = SGDClassifier(
            loss=self.loss,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

        # Hinge loss does not support predict_proba
        if self.loss == 'hinge':
            self.calibration_method = None
            self.probability_combiner = None
            base_model = base_sgd
        else:
            if self.calibration_method:
                base_model = CalibratedClassifierCV(base_estimator=base_sgd, method='sigmoid', cv=5)
            else:
                base_model = base_sgd

        # Build and fit the pipeline using safe training logic
        self.pipeline, self.probability_combiner = build_safe_pipeline(
            X_train_text=X_train_text,
            y_train=y_train,
            vectorizer=vectorizer,
            strategy_class=self.strategy_class,
            base_classifier=base_model,
            calibration_method=self.calibration_method,
            probability_combiner=self.probability_combiner
        )

    def predict(self, X_text):
        self.predictions = self.pipeline.predict(X_text)
        return self.predictions
        
    def evaluate(self, y_test):
        """
        Evaluate predictions using hierarchical precision, recall, F1, and per-level accuracy.
        """
        from sklearn.metrics import accuracy_score
        from hiclass.metrics import precision, recall, f1
        import pandas as pd

        if self.predictions is None:
            raise ValueError("Run predict() before evaluating.")

        predictions_df = pd.DataFrame(self.predictions, columns=y_test.columns)
        self.metrics_ = {
            'hierarchical_precision': precision(y_true=y_test, y_pred=predictions_df),
            'hierarchical_recall': recall(y_true=y_test, y_pred=predictions_df),
            'hierarchical_f1': f1(y_true=y_test, y_pred=predictions_df),
            'level_accuracy': {
                level: accuracy_score(y_test[level], predictions_df[level])
                for level in y_test.columns
            },
            'exact_path_accuracy': (predictions_df == y_test).all(axis=1).mean()
        }
        return self.metrics_

    def save(self, filepath):
        import joblib
        save_dict = {
            'pipeline': self.pipeline,
            'tfidf_params': self.tfidf_params,
            'strategy': self.strategy_class.__name__,
            'loss': self.loss,
            'max_iter': self.max_iter,
            'calibration_method': self.calibration_method,
            'probability_combiner': self.probability_combiner,
            'predictions': self.predictions,
            'evaluation_metrics': getattr(self, 'metrics_', None),
            'confidence_df': getattr(self, 'confidence_df', None)
        }
        joblib.dump(save_dict, filepath)
        print(f"✅ Model and metadata saved to {filepath}")

    def load(self, filepath):
        import joblib
        saved = joblib.load(filepath)
        self.pipeline = saved['pipeline']
        self.tfidf_params = saved['tfidf_params']
        self.strategy_class = eval(saved['strategy'])
        self.loss = saved['loss']
        self.max_iter = saved['max_iter']
        self.calibration_method = saved['calibration_method']
        self.probability_combiner = saved['probability_combiner']
        self.predictions = saved['predictions']
        self.metrics_ = saved.get('evaluation_metrics')
        self.confidence_df = saved.get('confidence_df')
        print(f"📦 Model loaded from {filepath}")

    def analyze_confidence_scores(self, X_test_text, y_test, top_n=5):
        import pandas as pd
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

        self.confidence_df = confidence_df

        avg_conf = confidence_df['Confidence'].mean()
        print(f"🔍 Average top confidence score: {avg_conf:.4f}")
        top_correct = confidence_df[confidence_df['Correct']].sort_values(by='Confidence', ascending=False).head(top_n)
        print(f"🎯 Top {top_n} most confident correct predictions:")
        print(top_correct)
        return confidence_df

    def get_confusion_matrices(self, y_test):
        from sklearn.metrics import confusion_matrix
        import pandas as pd

        predictions_df = pd.DataFrame(self.predictions, columns=y_test.columns)
        cm_dict = {}
        for level in y_test.columns:
            cm = confusion_matrix(y_test[level], predictions_df[level])
            cm_dict[level] = cm
        return cm_dict

    def print_confusion_matrices(self, y_test, label_names_dict=None, export_path=None):
        import pandas as pd
        from IPython.display import display
        from openpyxl import Workbook

        cm_dict = self.get_confusion_matrices(y_test)
        predictions_df = pd.DataFrame(self.predictions, columns=y_test.columns)

        writer = pd.ExcelWriter(export_path, engine='openpyxl') if export_path else None

        print("🧩 Confusion Matrices (per level):")
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

