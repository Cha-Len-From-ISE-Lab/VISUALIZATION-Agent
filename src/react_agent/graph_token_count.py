from react_agent import graph
from langchain_core.callbacks import get_usage_metadata_callback

def graph_invoke_with_token_count(system_prompt: str, user_message: str):
    with get_usage_metadata_callback() as cb:
        res = graph.invoke(
            {"messages": [("system", system_prompt), ("user", user_message)]},
            {"configurable": {"system_prompt": system_prompt}},
        )

    tokens = list(cb.usage_metadata.values())[0]
    model = list(cb.usage_metadata.keys())[0]

    return {
        "model": model,
        "response": res,
        "token": {
            "total": tokens["total_tokens"],
            "input": tokens["input_tokens"],
            "output": tokens["output_tokens"]
        }
    }

if __name__ == "__main__":
    system_prompt = "You are a dog, I'm a dog, let you message with me by dog's language."
    user_prompt = "gau gauw gwau aguw"
    tkc_res = graph_invoke_with_token_count(system_prompt, user_prompt)
    print(tkc_res)

    print(tkc_res["token"]["input"])
    print(tkc_res["token"]["output"])
    
    res = tkc_res["response"]
    print(str(res["messages"][-1].content).strip())


