from react_agent import graph

def _llm(system_prompt: str, user_message: str) -> str:
    res = graph.invoke(
        {"messages": [("system", system_prompt), ("user", user_message)]},
        {"configurable": {"system_prompt": system_prompt}},
    )
    return str(res["messages"][-1].content).strip()

def _detect_pre_risk(html_code: str) -> str:
    # system_prompt = "You are a senior front-end developer and code reviewer. "\
    #         "You analyze HTML files that include inline JavaScript and CSS. "\
    #         "Your task is to identify potential issues, especially in the JavaScript part, "\
    #         "based on the provided project description. You should consider common mistakes, "\
    #         "bad practices, and logic traps, but you must not assume that the code is definitively wrong yet. "\
    #         "Be critical but cautious, and point to possible risks."

    system_prompt = "You are a senior front-end developer and technical code reviewer. " \
        "You analyze HTML files that include inline JavaScript and CSS. " \
        "Your primary goal is to detect technical issues, especially related to JavaScript syntax, " \
        "variable handling, data structure misuse, and logic errors. " \
        "You do NOT need to focus on security or UX issues like CORS, accessibility, or best practices. " \
        "Concentrate strictly on potential programming mistakes or code that may cause runtime or logical errors." \
        "Only give me possible risk, no more message."

    user_message = "Below is a complete HTML file with embedded JavaScript and CSS.\n\n"\
        "Your task is to:\n"\
        "1. Analyze the code and identify potential JavaScript-related bugs or misuses.\n"\
        "2. Focus strictly on technical risks such as:\n   "\
        "- Syntax errors\n   "\
        "- Invalid variable usage or scoping problems\n   "\
        "- API response structure assumptions\n   "\
        "- Logic bugs or missing conditions\n   "\
        "- Misuse of DOM APIs or event handling\n   "\
        "- Misuse of async/await or unhandled promises\n"\
        "3. Ignore security, styling, CORS, accessibility, or general best practice suggestions.\n\n"\
        "### Full HTML Code:\n"\
        "```html"\
        f"{html_code}"
    
    return _llm(system_prompt, user_message)


def _detect_specific_bug(html_code: str, pre_risk: str, project_description: str, model_information: str, output_example: str) -> str:
    system_prompt = "You are a senior front-end engineer and code reviewer. Your job is to analyze an HTML file "\
        "that includes inline JavaScript and CSS, focusing especially on the JavaScript logic. "\
        "You are provided with a list of potential issues. Your task is to confirm whether each issue is real, "\
        "explain the reason, identify exact code locations, and suggest fixes if needed. "\
        "If the issue is not real, explain why it's safe or acceptable in context."
    
    user_message = "Below is a complete HTML file (with inline CSS and JavaScript), along with a list of potential issues "\
        "identified earlier based on the project description.\n\n"\
        "Your task:\n"\
        "1. Confirm whether each potential issue is an actual issue.\n"\
        "2. For real issues:\n"\
        "   - Classify the issue (e.g., logic, syntax, runtime, bad practice)\n"\
        "   - Point to specific code fragments or lines\n"\
        "   - Explain the issue\n"\
        "   - Suggest a concrete fix\n"\
        "3. For non-issues: explain why the code is valid.\n\n"\
        "Return your results as a structured list.\n\n"\
        "### Project Description:\n"\
        f"{project_description}\n\n"\
        "### Information of model (API endpoint):\n"\
        f"{model_information}"\
        f"Example output (response) of API (model)\n"\
        f"{output_example}"\
        "### List of Potential Issues:\n"\
        f"{pre_risk}\n\n"\
        "### Full Code:\n"\
        f"```html\n{html_code}\n```"
    
    return _llm(system_prompt, user_message)

def _fix_code(html_code: str, specific_bug: str, project_description: str) -> str:
    system_prompt = "You are a senior developer who specializes in debugging and refactoring front-end code. "\
        "You will be given an HTML file (with inline JavaScript and CSS), and a list of confirmed issues in it. "\
        "Your task is to fix the code based on these issues, without changing parts of the code that are correct. "\
        "The output must be a single fixed HTML file that works properly according to the project description."
    
    user_message = "### Project Description:\n"\
        f"{project_description}\n\n"\
        "### Confirmed Issues:\n"\
        f"{specific_bug}\n\n"\
        "### Original Code:\n"\
        f"```html\n{html_code}\n```\n\n"\
        "Please return the corrected full HTML code below:"
    
    return _llm(system_prompt, user_message)

# Use this function only
def detect_bug_n_fix(html_code: str, project_description: str, model_information: str, output_example: str) -> str:
    pre_risk = _detect_pre_risk(html_code)
    specific_bug = _detect_specific_bug(html_code, pre_risk, project_description, model_information, output_example)
    fixed_code = _fix_code(html_code, specific_bug, project_description)
    if fixed_code.startswith("```html"):
        fixed_code = fixed_code.strip("```html").strip("```").strip()
    
    return fixed_code


if __name__ == "__main__":
    from yaml_extracter import extract_description, extract_model_info
    yaml_path = "./_test_extracter/task.yaml"
    with open("./_test_checker/test_html.html", "r", encoding="utf-8") as f:
        code = f.read()
        print(code)
        project_description = extract_description(yaml_path)
        model_information = extract_model_info(yaml_path)
        output_example = "unknown"
        fixed_code = detect_bug_n_fix(code, project_description, model_information, output_example)
        print(fixed_code)
        with open("./_test_checker/fixed_test_html.html", "w", encoding="utf-8") as f:
            f.write(fixed_code)

