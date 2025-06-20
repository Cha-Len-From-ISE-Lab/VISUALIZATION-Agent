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
from yaml_extracter import *

def specification_agent(task: str):
    system_prompt = """
You are an expert in AI Engineering and UI/UX Design, assisting in a UI auto-generation competition.

Each task comes with a `task.yaml` file describing the problem type, expected input/output formats, model APIs, interaction requirements, and sometimes data visualization needs.

Your goal is to analyze this task definition and generate a comprehensive UI specification that will be used to guide interface generation. Your output must include the following three sections:

### 1. HTML_SPEC
Describe the structure of the interface, broken down into clearly defined functional areas. These may include (but are not limited to):
- **Unit Processing Area**: for single-instance input, prediction, and result visualization
- **Batch Processing Area**: for file upload, batch execution, and result display
- **Data Visualization Area**: for rendering input tables, highlighting predictions, showing annotations, or reviewing answers
- **Instructional or Output Summary Panel**: for additional interpretation or summarization
- **History or Query Review Area**: for maintaining prior queries or interactions

These sections should be determined based on the specific requirements of the task, such as:
- Input/output format (text, image, table, JSON)
- Interaction modality (single-step vs multi-step)
- Visualization instructions (e.g., cell highlighting, answer coordinates)
- Model capabilities and API structure

For each area, describe:
- Required inputs
- Output or display elements
- Layout arrangement (e.g., side-by-side, vertical stack)
- Logical grouping of elements

### 2. CSS_SPEC
Describe the intended appearance and layout style, including:
- Spacing, alignment, and content grouping
- Visual separation between functional sections
- Responsive design (mobile/desktop)
- Color palette usage (choose one consistent Tailwind color family)
- Readability and accessibility guidelines (contrast, font size, hover/focus states)
- Avoiding components shrinking to fit their content only; use `min-w-`, `min-h-`, `w-full`, `h-64`, etc. where appropriate.

### 3. JS_SPEC
Describe the interactive behaviors, including:
- Input handling (form input, file uploads, validations)
- Data transformation logic (e.g., parsing TSV, mapping table coordinates)
- API interaction (fetch calls, error handling, state updates)
- Dynamic rendering (e.g., updating DOM with results, drawing highlight overlays)
- Optional features such as:
    - History saving/loading (via localStorage or in-memory)
    - Visualization overlays (e.g., coloring cells)
    - Pagination or scroll areas for long batch outputs

Use abstract description only—no code should be included. Your specification must adapt to the task’s data type, interaction complexity, and visualization needs.
"""

    user_prompt = f'''
Below is a `task.yaml` file that defines the task. Please analyze it and generate the UI specification as instructed.
{task}

Your response must contain three sections:

### HTML_SPEC
Clearly describe the full interface structure, broken into logical sections that match the nature of the task. Each section must include inputs, outputs, layout intentions, and their role.

### CSS_SPEC
Describe the visual design choices for the layout, spacing, contrast, responsiveness, and theme consistency.

### JS_SPEC
Describe the functional logic that connects user input with backend inference and updates the display dynamically. Include batch handling, parsing, validation, and visualization logic if needed.
'''
    return system_prompt, user_prompt




def design_agent(html_spec: str, css_spec: str, js_spec: str) -> tuple[str, str]:
    system_prompt = """
You are a Senior UI/UX Architect.

Your role is to take UI specifications—including HTML layout description, CSS styling intent, and JavaScript behavior—and synthesize them into a **comprehensive high-level design plan**.

This design plan should reflect the structure and flow of the user interface, without going into low-level code. Think of it as the output of the "Design" phase in the Waterfall model.

The design must include:

1. **Component Map**: List and describe all major UI components (e.g., Header, Input Form, Model Output Panel, Query History Section, etc.).
2. **Page Layout Structure**: Describe the layout hierarchy, grouping, and spatial arrangement. Include ideas like sections, columns, cards, modals, tabs, etc.
3. **User Interaction Flow**: Explain how users will interact with the interface from input to output. Include what triggers what, and how the flow is reflected in UI structure.
4. **Responsiveness and Accessibility**: Mention layout behavior on different screen sizes, accessibility considerations (keyboard, screen reader, contrast, etc.).
5. **Design Considerations**: Highlight design trade-offs, assumptions, or open questions (e.g., when to use modal vs new section; if query history is expandable, etc.).

**Important**: If any output section (like batch results) can grow too long due to user interaction or data size, suggest UX strategies such as scrollable containers with fixed max height or pagination mechanisms to ensure layout remains manageable and user-friendly.

Do NOT include any code or HTML/CSS/JS syntax. Focus only on the conceptual structure and user experience design.
"""
    user_prompt = f"""
Here is the UI specification you need to analyze:

### HTML_SPEC
{html_spec}

### CSS_SPEC
{css_spec}

### JS_SPEC
{js_spec}

Based on the above, generate a complete high-level design plan as described.
"""
    return system_prompt, user_prompt


def html_generator_agent(html_spec: str, design: str) -> tuple[str, str]:
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

Generate the full HTML code based strictly on the given UI specification and design layout description.
"""

    user_prompt = f"""
Below is the UI specification (HTML_SPEC) describing the required components of the interface:

{html_spec}

In addition, here is the high-level design plan (DESIGN) describing the overall layout, component structure, and interaction flow:

{design}

Using both the HTML_SPEC and DESIGN, generate a full and well-structured HTML5 file that reflects both the required elements and their correct arrangement within the page layout.
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

Output of ML task always is the label with highest probability.

Please return the complete HTML with JavaScript inserted at the end.
"""
    return system_prompt, user_prompt

def tailwind_styler_agent(html_code: str, css_spec: str, design: str) -> tuple[str, str]:
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
- **Avoid UI blocks that shrink to fit only their content**. Use utilities like `min-w-`, `min-h-`, `flex-1`, or fixed width/height classes (`w-full`, `h-64`, etc.) to ensure layout stability.

2. **Accessibility & Responsiveness**
- Use appropriate colors, contrast, and sizes for accessibility.
- Ensure the layout works well on both desktop and mobile (e.g., responsive widths, stacking, spacing).
- Keep interaction elements keyboard-friendly.

3. **Technical Notes**
- Do not modify the semantic structure of HTML.
- Do not add traditional CSS or <style> blocks.
- Inject Tailwind utility classes into `class=""` attributes.
- Make sure to include the Tailwind CDN link in the <head> of the HTML.

4. **Color Palette Consistency**
- Choose and stick to a single Tailwind color family (e.g., blue, emerald, amber, slate...) for primary UI components such as buttons, borders, highlights, and headings.
- Apply different shades (e.g., `blue-100`, `blue-500`, `blue-700`) from that color family across the interface for visual hierarchy and contrast.
- Do not mix multiple unrelated color families (e.g., don’t combine blue with red or green).
- Ensure text remains readable on colored backgrounds using proper contrast (e.g., `text-white` on `bg-blue-600`).

Return the full updated HTML file with Tailwind classes applied directly to elements.
"""


    user_prompt = f"""
Here is the HTML file:
{html_code}

Here is the CSS_SPEC describing the desired appearance:
{css_spec}

And here is the high-level DESIGN document that explains the layout structure and component roles:
{design}

Please use both the CSS_SPEC and DESIGN document to apply appropriate Tailwind classes for layout, appearance, and responsiveness.

**Important**: Some containers or cards in the interface (such as the input box, result section, or history panel) may shrink too much if the content is small. Prevent this behavior by using appropriate Tailwind classes to maintain a consistent and readable layout. Use `min-width`, `min-height`, or `flex-grow` utilities to avoid awkward collapsing.
**Important**: If any output section (like batch results) can grow too long due to user interaction or data size, suggest UX strategies such as scrollable containers with fixed max height or pagination mechanisms to ensure layout remains manageable, consistency and user-friendly.

Make sure the final HTML file includes:
- Tailwind classes added into each relevant `class=""` attribute
- Visual hierarchy, spacing, layout, and interaction states per CSS_SPEC
- The Tailwind CDN link added inside the <head> section
- Strictly no external or internal <style> blocks
"""
    return system_prompt, user_prompt



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

def generate_fe(task_path:str) -> None:
    with open(task_path, 'r', encoding='utf-8') as f:
        task_info = yaml.safe_load(f)
    
    system_prompt, user_prompt = specification_agent(str(task_info))
    res = graph.invoke(
        {"messages": [("system", system_prompt), ("user", user_prompt)]},
        {"configurable": {"system_prompt": system_prompt}},
    )
    
    html_spec, css_spec, js_spec = extract_specs(str(res["messages"][-1].content).strip())
    
    system_prompt, user_prompt = design_agent(html_spec, css_spec, js_spec)
    design_res = graph.invoke(
        {"messages": [("system", system_prompt), ("user", user_prompt)]},
        {"configurable": {"system_prompt": system_prompt}},
    )
    design = str(design_res["messages"][-1].content).strip()

    html_system, html_user = html_generator_agent(html_spec, design)
    html_res = graph.invoke(
        {"messages": [("system", html_system), ("user", html_user)]},
        {"configurable": {"system_prompt": html_system}},
    )
    html_code = str(html_res["messages"][-1].content).strip()

    api_url = task_info.get("model_information", {}).get("api_url", "")
    input_example, output_example = get_model_output(task_path)
    
    js_system, js_user = js_injector_agent(html_code, js_spec, api_url, input_example, output_example)
    js_res = graph.invoke(
        {"messages": [("system", js_system), ("user", js_user)]},
        {"configurable": {"system_prompt": js_system}},
    )
    html_with_js = str(js_res["messages"][-1].content).strip()


    css_system, css_user = tailwind_styler_agent(html_with_js, css_spec, design)
    css_res = graph.invoke(
        {"messages": [("system", css_system), ("user", css_user)]},
        {"configurable": {"system_prompt": css_system}},
    )
    final_html = str(css_res["messages"][-1].content).strip()


    with open("tests/integration_tests/generated_ui.html", "w", encoding="utf-8") as f:
        f.write(final_html.replace("```html", "").replace("```", ""))
    print("Saved to file tests/integration_tests/generated_ui.html")

    project_description = extract_description(task_path)
    model_information = extract_model_info(task_path)
    
    fixed_code = detect_bug_n_fix(final_html, project_description, model_information, input_example, output_example)
    
    with open("tests/integration_tests/fixed_generated_ui.html", "w", encoding="utf-8") as f:
        f.write(fixed_code)
        print("Saved to file tests/integration_tests/fixed_generated_ui.html")

if __name__ == "__main__":
    generate_fe("src/react_agent/task (3).yaml")