from utils import get_model_output
from utils import extract_format_from_yaml
import yaml

# Load YAML file
with open('D:\ISE - Challenges\VISUALIZATION-Agent\\text_classification_verified\\task.yaml', 'r', encoding='utf-8') as file:
    yaml_description = yaml.safe_load(file)

api_url = "http://34.142.220.207:8000/api/text-classification"

# Test with sample input data
output = get_model_output(api_url=api_url, yaml_description=yaml_description)
print(output)
print(type(output))