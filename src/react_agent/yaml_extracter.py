import yaml

def extract_info(field: str, yaml_src: str):
    def get_nested(data, keys):
        for key in keys:
            if isinstance(data, list):
                try:
                    key = int(key)
                    data = data[key]
                except (ValueError, IndexError):
                    return None
            elif isinstance(data, dict):
                data = data.get(key)
            else:
                return None
        return data

    with open(yaml_src, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    keys = field.split('.')
    return get_nested(data, keys)

def extract_description(yaml_src: str):
    return extract_info("task_description.description", yaml_src)

def extract_model_info(yaml_src: str):
    return extract_info("model_information", yaml_src)

if __name__ == "__main__":
    yaml_path = "./_test_extracter/task.yaml"
    print(extract_description(yaml_path))
    print(extract_model_info(yaml_path))

    yaml_path = "./_test_extracter/task1.yaml"
    print(extract_description(yaml_path))
    print(extract_model_info(yaml_path))


