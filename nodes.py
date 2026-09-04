from langgraph.graph import END, StateGraph, MessagesState
# from schemas import Documents


class State(MessagesState):
    documents = list[str]
    query: str
    llm_response: str
    searched: int
    scope: str


from chains import initial_chain, comparison_chain, response_chain, tools, scope_chain

INITIAL = "initial"
COMPARISON = "comparison"
TOOLS = "tools"
RESPONSE = "response"
SCOPE = "scope"
DECLINE = "decline"

def scope_node(state: State):
    res = scope_chain.invoke({"query": state["query"]})
    return {"scope": res.scope}


def accepted_or_declined(state: State):
    if state["scope"]:
        return INITIAL
    else:
        return DECLINE

def decline_node(state: State):
    return {"llm_response": "Sorry, this query is not in my scope."}


def initial_node(state: State):
    result = initial_chain.invoke({"query": state["query"]})
    return {"messages": [result]}

def comparison_node(state: State):
    result = comparison_chain.invoke({"query": state["query"], "documents": state["messages"][-1].content})
    return {"messages": [result]}


def tool_node(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls and state["searched"] < 4:
        state["searched"] = state.get("searched", 0) + 1
        return TOOLS
    return RESPONSE

def response_node(state: State):
    result = response_chain.invoke({"query": state["query"], "documents": state["messages"][-1].content})
    return {"llm_response": result.content}


build = StateGraph(state_schema=State)
build.add_node(SCOPE, scope_node)
build.set_entry_point(SCOPE)
build.add_node(DECLINE, decline_node)
build.add_conditional_edges(SCOPE, accepted_or_declined, path_map= {INITIAL: INITIAL, DECLINE: DECLINE})
build.add_edge(DECLINE, END)
build.add_node(INITIAL, initial_node)
build.add_node(TOOLS, tools)
build.add_conditional_edges(INITIAL, tool_node, path_map= {TOOLS: TOOLS, RESPONSE: RESPONSE})
build.add_node(COMPARISON, comparison_node)
build.add_edge(TOOLS, COMPARISON)
build.add_node(RESPONSE, response_node)
build.add_conditional_edges(COMPARISON, tool_node, path_map = {TOOLS: TOOLS, RESPONSE: RESPONSE})

graph = build.compile()



if __name__ == "__main__":
    graph.get_graph().draw_mermaid_png(output_file_path = "graph.png")





