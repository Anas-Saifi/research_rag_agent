from langgraph.graph import END, StateGraph, MessagesState
# from schemas import Documents


class State(MessagesState):
    documents = list[str]
    query: str
    llm_response: str
    searched: int


from chains import initial_chain, comparison_chain, response_chain, tools

INITIAL = "initial"
COMPARISON = "comparison"
TOOLS = "tools"
RESPONSE = "response"

def initial_node(state: State):
    result = initial_chain.invoke({"query": state["query"]})
    return {"messages": [result]}

def comparison_node(state: State):
    result = comparison_chain.invoke({"query": state["query"], "documents": state["messages"][-1].content})
    return {"messages": [result]}


def tool_node(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls and state["searched"] < 4:
        state.get("searched", 0) + 1
        return TOOLS
    return RESPONSE

def response_node(state: State):
    result = response_chain.invoke({"query": state["query"], "documents": state["messages"][-1].content})
    return {"llm_response": result.content}


build = StateGraph(state_schema=State)
build.add_node(INITIAL, initial_node)
build.set_entry_point(INITIAL)
build.add_node(TOOLS, tools)
build.add_edge(INITIAL, TOOLS)
build.add_node(COMPARISON, comparison_node)
build.add_edge(TOOLS, COMPARISON)
build.add_node(RESPONSE, response_node)
build.add_conditional_edges(COMPARISON, tool_node, path_map = {TOOLS: TOOLS, RESPONSE: RESPONSE})

graph = build.compile()



if __name__ == "__main__":
    graph.get_graph().draw_mermaid_png(output_file_path = "graph.png")





