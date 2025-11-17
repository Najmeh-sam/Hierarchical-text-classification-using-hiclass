import sys
# Adding the relevant folders to the python path
sys.path.append("./modules")

from helper.display_utils import build_model_configuration_widgets
from IPython.display import display, HTML

display(HTML("<h2>🛠️ Model Configuration Inputs</h2><p>Please fill in the configuration below to build and train your HiClass model.</p>"))
# print("Please fill in the configuration below to build and train your HiClass model.")

dynamic_values = build_model_configuration_widgets()
