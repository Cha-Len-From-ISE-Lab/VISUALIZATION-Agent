import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.react_agent import graph

import pytest
import asyncio
from langsmith import unit
from react_agent import graph
from checker import detect_bug_n_fix
import yaml
from src.react_agent.utils import get_model_output
import requests

def receptionist_agent(task: str):
    system_prompt = """
You are an expert in both AI Engineering and UI/UX Design. You are assisting in a UI auto-generation competition.
The goal of the competition is to develop a system that can automatically generate interactive user interfaces (UIs) from given task specifications. Each task comes in a `task.yaml` file, describing the goal, model, input/output data, and user interaction requirements.

Your job is to analyze the task and generate a **UI specification** that clearly defines:
1. What should be included in the HTML structure (elements like forms, buttons, sections for model output, file upload areas, etc.).
2. What kind of CSS styling is necessary to make the interface intuitive, usable, and visually pleasant.
3. What JavaScript logic is needed to make the interface interactive (such as handling input, calling the model API, rendering model results, etc.).

The output must be three clear and separate sections: 
- HTML_SPEC
- CSS_SPEC
- JS_SPEC

These specifications will be passed to specialized code-generation agents to create the actual UI files. So be precise, and do not include actual code—only descriptions of structure, styling intent, and logic behavior.
"""
    user_prompt = f'''
Below is a `task.yaml` file that defines the task. Please analyze it and generate the UI specification as instructed.
{task}
Your response must contain three sections:

### HTML_SPEC
Describe the required layout and elements. Be clear about what each part of the interface is for, what inputs are needed, and how users interact with the interface.

### CSS_SPEC
Describe the styling of each UI component. Consider layout, spacing, colors, fonts, hover/focus states, and responsiveness.

### JS_SPEC
Describe the logic needed to handle user input, communicate with the model (API calls or local inference), process responses, and update the interface accordingly.
'''
    return system_prompt, user_prompt


def html_generator_agent(html_spec: str) -> str:
    system_prompt = """
You are a professional frontend engineer. Your task is to generate a clean, semantic, and accessible HTML5 file from a given UI specification.

The HTML must:
- Use proper tags like <form>, <label>, <input>, <section>, <button>, etc.
- Include semantic structure (e.g., <header>, <main>, <footer> if appropriate)
- Be readable, properly indented, and commented
- Contain placeholder IDs and class names that will be used later for styling and JavaScript logic
- Include minimal inline styles (preferably none)
- Be self-contained (including <!DOCTYPE html>, <html>, <head>, <body>)

**DO NOT** include any CSS or JavaScript inside this HTML file — it will be handled separately.

Generate the full HTML code based strictly on the given UI specification.
"""
    
    user_prompt = f"""
Below is the UI specification (HTML_SPEC) describing what the interface should contain. Generate a complete HTML5 file accordingly.
{html_spec}
"""
    
    return system_prompt, user_prompt

def js_injector_agent(html_code: str, js_spec: str, api_url: str, input_example: str, output_example: str) -> str:
    system_prompt = f"""
You are a senior frontend developer. Enhance the provided HTML file by adding JavaScript functionality as described in the JS_SPEC.

Requirements:
- Analyze the HTML and JS_SPEC carefully.
- Write a <script> block to add all required interactivity, ensuring a smooth and user-friendly experience.
- Insert the <script> block at the end of the <body>.
- Use clear, readable vanilla JavaScript (no frameworks).
- Use IDs and classes from the HTML for element selection.
- Do not alter the existing HTML structure unless absolutely necessary.
- Ensure all user interactions are smooth, with proper error handling and UI updates.

To handel Machine Learning task please abide by the following format:
API endpoint: {api_url}
Example of API input format:
{input_example}
Example of API output format:
{output_example}

Return the full updated HTML file with the JavaScript embedded.
"""
    user_prompt = f"""
Here is the HTML file:
{html_code}

And here is the JS_SPEC describing the logic to be implemented:

{js_spec}

Please return the complete HTML with JavaScript inserted at the end.
"""
    return system_prompt, user_prompt

def tailwind_styler_agent(html_code: str, css_spec: str) -> tuple[str, str]:
    system_prompt = """
You are a UI/UX expert and Tailwind CSS specialist.

Your task is to enhance an existing HTML file by applying Tailwind CSS utility classes to make it visually appealing, user-friendly, and accessible.

Instead of writing traditional CSS, use Tailwind utility classes directly in the HTML elements by adding or editing the `class=""` attribute.

Your responsibilities:
1. **Readability & Usability**
- Apply spacing (margin/padding), layout (flex/grid), and typography utilities for clarity and good structure.
- Style form elements, buttons, input fields, and result displays so they are easy to read and interact with.
- Add visual feedback (e.g., `hover:`, `focus:`, `disabled:` states) where appropriate.
- Add headings, labels, buttons, and containers with clear visual hierarchy.

2. **Accessibility & Responsiveness**
- Use appropriate colors, contrast, and sizes for accessibility.
- Ensure the layout works well on both desktop and mobile (e.g., responsive widths, stacking, spacing).
- Keep interaction elements keyboard-friendly.

3. **Technical Notes**
- Do not modify the semantic structure of HTML.
- Do not add traditional CSS or <style> blocks.
- Inject Tailwind utility classes into `class=""` attributes.
- Make sure to include the Tailwind CDN link in the <head> of the HTML.

Return the full updated HTML file with Tailwind classes applied directly to elements.
"""

    user_prompt = f"""
Here is the HTML file:
{html_code}

And here is the CSS_SPEC describing the desired appearance:

{css_spec}

Please return the complete HTML file with Tailwind classes added directly in the HTML elements.
Make sure to include the Tailwind CDN link in the <head>.
"""

    return system_prompt.strip(), user_prompt.strip()

def extract_specs(output):
    html_marker = "HTML_SPEC"
    css_marker = "CSS_SPEC"
    js_marker = "JS_SPEC"

    html_start = output.find(html_marker)
    css_start = output.find(css_marker)
    js_start = output.find(js_marker)

    if html_start == -1 or css_start == -1 or js_start == -1:
        raise ValueError("Không tìm thấy đủ 3 section trong output")

    html_spec = output[html_start + len(html_marker):css_start].strip("- \n:")
    css_spec = output[css_start + len(css_marker):js_start].strip("- \n:")
    js_spec = output[js_start + len(js_marker):].strip("- \n:")

    return html_spec.strip(), css_spec.strip(), js_spec.strip()

def extract_project_info_from_task_info(task_info: str):
    data = yaml.safe_load(task_info)

    project_description = data.get('task_description', {}).get('description', '')

    input_format = data.get('model_information', {}).get('input_format', {})
    if isinstance(input_format, dict):
        api_input_format = yaml.dump(input_format, allow_unicode=True)
    else:
        api_input_format = str(input_format)

    output_format = data.get('model_information', {}).get('output_format', {})
    if isinstance(output_format, dict):
        api_output_example = yaml.dump(output_format, allow_unicode=True)
    else:
        api_output_example = str(output_format)
    return project_description, api_input_format, api_output_example

def generate_fe(task_path) -> None:
    with open(task_path, 'r', encoding='utf-8') as f:
        task_info = yaml.safe_load(f)
    
    system_prompt, user_prompt = receptionist_agent(str(task_info))
    res = graph.invoke(
        {"messages": [("system", system_prompt), ("user", user_prompt)]},
        {"configurable": {"system_prompt": system_prompt}},
    )
    
    html_spec, css_spec, js_spec = extract_specs(str(res["messages"][-1].content).strip())
    print(html_spec)
    print(css_spec)
    print(js_spec)

    html_system, html_user = html_generator_agent(html_spec)
    html_res = graph.invoke(
        {"messages": [("system", html_system), ("user", html_user)]},
        {"configurable": {"system_prompt": html_system}},
    )
    html_code = str(html_res["messages"][-1].content).strip()

    api_url = task_info.get("model_information", {}).get("api_url", "")
    input_example, output_example = get_model_output(task_path)
    
    js_system, js_user = js_injector_agent(html_code, js_spec, api_url, input_example, output_example)
    print("JS System", js_system)
    print("JS User", js_user)
    js_res = graph.invoke(
        {"messages": [("system", js_system), ("user", js_user)]},
        {"configurable": {"system_prompt": js_system}},
    )
    html_with_js = str(js_res["messages"][-1].content).strip()


    css_system, css_user = tailwind_styler_agent(html_with_js, css_spec)
    css_res = graph.invoke(
        {"messages": [("system", css_system), ("user", css_user)]},
        {"configurable": {"system_prompt": css_system}},
    )
    final_html = str(css_res["messages"][-1].content).strip()


    with open("tests/integration_tests/generated_ui.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("Saved to file tests/integration_tests/generated_ui.html")

    project_description, api_input_format, api_output_example = extract_project_info_from_task_info(task_info)
    
    fixed_code = detect_bug_n_fix(final_html, project_description, api_input_format, api_output_example)
    
    with open("tests/integration_tests/fixed_generated_ui.html", "w", encoding="utf-8") as f:
        f.write(fixed_code)
        print("Saved to file tests/integration_tests/fixed_generated_ui.html")

if __name__ == "__main__":
    generate_fe("src/react_agent/task (1).yaml")