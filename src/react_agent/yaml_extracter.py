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

def extract_model_input(yaml_src: str):
    return extract_info("model_information.input_format", yaml_src)

if __name__ == "__main__":
    yaml_path = "./_test_extracter/task.yaml"
    print(extract_description(yaml_path))
    print(extract_model_info(yaml_path))

    yaml_path = "./_test_extracter/task.yaml"
    print(extract_description(yaml_path))
    print(extract_model_info(yaml_path))

    print(extract_model_input(yaml_path))

    from react_agent import graph

    sys_prt = "You are a JSON input generator for machine learning APIs. You will receive an `input_format` specification describing " \
        "the expected structure of a JSON input."\
        "Your job is to generate a valid sample JSON input based on the given format."\
        "Guidelines:"\
        "- Only return the JSON input. Do not include explanations or descriptions."\
        "- Fill in realistic and coherent dummy data for each field, based on field names and types."\
        "- Always match the required structure and types exactly."\
        "Never repeat the schema or format. Only output the resulting JSON input."
    
    user_prompt = "Based on the following input_format schema, generate a valid JSON input:"\
        f"{extract_model_input(yaml_path)}"

    res = graph.invoke(
        {"messages": [("system", sys_prt), ("user", user_prompt)]},
        {"configurable": {"system_prompt": sys_prt}},
    )
    print('\n\n')
    print(str(res["messages"][-1].content).strip())

    import requests
    import json
    
    url = 'http://34.142.220.207:8000/api/text-classification'
    payload = json.loads(str(res["messages"][-1].content).strip())
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Response JSON:", response.json())
    else:
        print("Request failed:", response.status_code)

    




