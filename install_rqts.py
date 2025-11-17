#Check the requirements file to ensure all necessary libraries are installed
import sys
sys.path.append("./helper")
from helper import install_requirements #helper.helper
# find the requirements.txt file in the main folder
install_requirements('./requirements.txt')
