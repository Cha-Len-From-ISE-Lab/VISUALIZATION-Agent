from react_agent import graph
from langchain_community.callbacks import get_openai_callback

def graph_invoke_with_token_count(system_prompt: str, user_message: str):
    with get_openai_callback() as cb:
        res = graph.invoke(
            {"messages": [("system", system_prompt), ("user", user_message)]},
            {"configurable": {"system_prompt": system_prompt}},
        )

    return {
        "response": res,
        "token": {
            "total": cb.total_tokens,
            "input": cb.prompt_tokens,
            "output": cb.completion_tokens
        },
        "cost": cb.total_cost
    }

if __name__ == "__main__":
    system_prompt = "You are a dog, I'm a dog, let you message with me by dog's language."
    user_prompt = "gau gauw gwau aguw"
    print(graph_invoke_with_token_count(system_prompt, user_prompt))

