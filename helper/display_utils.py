# -----------------------------------------------------------------------------
# display_utils.py
#
# Author: Najmeh Samadiani <najmeh.samadiani@abs.gov.au>
# Created: 2025
# Description: Interactive widget builders and visualization utilities for
#              configuring and displaying results of hierarchical classifiers.
# -----------------------------------------------------------------------------

"""
display_utils.py

This module provides user interface components and display utilities for
interacting with hierarchical classification pipelines in Jupyter notebooks.

Included Components:
- `build_model_configuration_widgets()`: Dynamically builds and tracks interactive
  ipywidgets for selecting dataset inputs, model types, TF-IDF options, and HiClass settings.
- `plot_category_counts()`: Plots value counts of a selected categorical level using a collapsible bar chart.
- `display_code_differences_between_sets()`: Compares and visualizes category codes
  found in training vs test data using collapsible HTML blocks.
- `show_confidence_summary()`: Displays average model confidence and lists top confident predictions.

These tools are meant to support exploratory analysis, interactive experimentation,
and educational demonstrations of hierarchical classification models.
"""

from IPython.display import display, HTML
import ipywidgets as widgets
import matplotlib.pyplot as plt
import io
import base64

# --- Function to Build and Track Interactive Widgets for HiClass Configuration ---
# Includes dataset selector and toggles for conditional display

def build_model_configuration_widgets():
    
    def labeled_widget(widget, label):
        return widgets.VBox([widgets.HTML(f"<b>{label}</b>"), widget])

    dynamic_values = {}

    def update_value(change, key):
        dynamic_values[key] = change['new']

    def track_widget(key, widget):
        widget.observe(lambda change: update_value(change, key), names='value')
        return widget

    # Define dataset selector and toggle cell containers
    dataset_selector = widgets.Dropdown(
        options=['dataset1', 'dataset2'],
        value='dataset1',
        description='Dataset'
    )
    
    track_widget('dataset', dataset_selector)
    dynamic_values['dataset'] = dataset_selector.value

    # Define widgets
    widgets_dict_data = {
        'text_column': widgets.Dropdown(options=['Title', 'Text', 'Both'], value='Title'),
        'min_df': widgets.IntText(value=1),
        'max_df': widgets.FloatText(value=0.5),
        'max_features': widgets.IntText(value=50000),
        'stop_words': widgets.Dropdown(options=['english', None], value='english'),
        'ngram_min': widgets.IntText(value=1),
        'ngram_max': widgets.IntText(value=2),
    }

    widgets_dict_settings={
        'Hiclass_strategy': widgets.Dropdown(options=['lcppn', 'lcpn', 'lcpl'], value='lcppn'),
        'SGD_loss': widgets.Dropdown(options=['hinge', 'log_loss'], value='hinge'),
        'SGD_class_weight': widgets.Dropdown(options=['balanced', None], value='balanced'),
        'SGD_max_iter': widgets.IntText(value=1000),
        'Random_state': widgets.IntText(value=0),

        'RF_estimators': widgets.IntText(value=100),
        'RF_max_depth': widgets.IntText(value=None),
        'RF_criterion': widgets.Dropdown(options=['gini', 'entropy'], value='gini'),
    }

    for key, widget in widgets_dict_data.items():
        track_widget(key, widget)
        dynamic_values[key] = widget.value

    for key, widget in widgets_dict_settings.items():
        track_widget(key, widget)
        dynamic_values[key] = widget.value

    # External widgets
    base_classifier_widget = widgets.SelectMultiple(
        options=['SGD', 'LogisticRegression', 'RandomForest'],
        value=['SGD', 'LogisticRegression', 'RandomForest']
    )
    track_widget('base_classifiers', base_classifier_widget)
    dynamic_values['base_classifiers'] = base_classifier_widget.value

    calib_prob_dropdown = widgets.Dropdown(
        options=['none', 'probability only', 'calibration only', 'both'],
        value='both'
    )
    track_widget('calib_prob_control', calib_prob_dropdown)
    dynamic_values['calib_prob_control'] = calib_prob_dropdown.value

    calibration_method_dropdown = widgets.Dropdown(
        options=['isotonic', 'beta', 'ivap', 'cvap'], value='isotonic'
    )
    probability_combiner_dropdown = widgets.Dropdown(
        options=['geometric', 'multiply', 'arithmetic'], value='geometric'
    )

    # Containers
    calib_prob_container = widgets.VBox([
        labeled_widget(calibration_method_dropdown, "Calibration Method"),
        labeled_widget(probability_combiner_dropdown, "Probability Combiner")
    ])
    calib_prob_container.layout.display = 'block' if calib_prob_dropdown.value != 'none' else 'none'

    def toggle_calib_prob_visibility(change):
        calib_prob_container.layout.display = 'block' if change['new'] != 'none' else 'none'

    calib_prob_dropdown.observe(toggle_calib_prob_visibility, names='value')
    
    # --- Label column input per dataset ---
    label_inputs = {
        'dataset1': widgets.Text(value='Cat1,Cat2,Cat3', description='Labels for dataset1'),
        'dataset2': widgets.Text(value='coicop_label', description='Labels for dataset2')
    }
    label_box_container = widgets.VBox([label_inputs['dataset1']])

    def update_label_input_visibility(change):
        label_box_container.children = [label_inputs[change['new']]]

    dataset_selector.observe(update_label_input_visibility, names='value')

    # Display dataset selector and context
    display(labeled_widget(dataset_selector, "Choose Dataset"))
    display(label_box_container)

    # Display widgets
    for key, widget in widgets_dict_data.items():
        display(labeled_widget(widget, key.replace('_', ' ').title()))

    display(labeled_widget(base_classifier_widget, "Select Base Classifiers (multi-select)"))
    display(labeled_widget(calib_prob_dropdown, "Calibration/Probability Control"))
    display(calib_prob_container)

    for key, widget in widgets_dict_settings.items():
        display(labeled_widget(widget, key.replace('_', ' ').title()))
        
    # Register new dropdowns manually
    track_widget('calibration_method', calibration_method_dropdown)
    dynamic_values['calibration_method'] = calibration_method_dropdown.value

    track_widget('probability_combiner', probability_combiner_dropdown)
    dynamic_values['probability_combiner'] = probability_combiner_dropdown.value

    # Track label column values
    for key, widget in label_inputs.items():
        dynamic_values[f'label_columns_{key}'] = [col.strip() for col in widget.value.split(',') if col.strip()]
        widget.observe(lambda change, k=key: dynamic_values.__setitem__(f'label_columns_{k}', [c.strip() for c in change['new'].split(',')]), names='value')
    
    return dynamic_values

# --- Expandable Plot Renderer ---
def plot_category_counts(series, title, cat_num):
    counts = series.value_counts()
    if cat_num == 3:
        counts = counts.head(80)

    fig, ax = plt.subplots(figsize=(30, 4))
    counts.plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=70, labelsize=9)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    html = f"""
    <details>
        <summary><strong>{title}</strong></summary>
        <img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}" />
    </details>
    """
    display(HTML(html))

# --- Code Difference Viewer ---
def display_code_differences_between_sets(train_flat_cat, test_flat_cat):
    missing_in_test = set(train_flat_cat.unique()) - set(test_flat_cat.unique())
    missing_in_train = set(test_flat_cat.unique()) - set(train_flat_cat.unique())

    display(HTML(f'''
    <h3>➕ Codes in Training but not in Test ({len(missing_in_test)} codes)</h3>
    <details><summary>Click to expand</summary>
    <pre>{chr(10).join(str(x) for x in sorted(missing_in_test))}</pre>
    </details>

    <h3>➕ Codes in Test but not in Training ({len(missing_in_train)} codes)</h3>
    <details><summary>Click to expand</summary>
    <pre>{chr(10).join(str(x) for x in sorted(missing_in_train))}</pre>
    </details>
    '''))


# --- Confidence Summary Helper ---
def show_confidence_summary(confidence_df, top_n=5):
    avg_conf = confidence_df['Confidence'].mean()
    display(HTML(f"<b>🔍 Average top confidence score:</b> {avg_conf:.4f}"))

    top_correct = confidence_df[confidence_df['Correct']].sort_values(by='Confidence', ascending=False).head(top_n)
    display(HTML(f"<b>🎯 Top {top_n} most confident correct predictions:</b>"))
    display(top_correct)
