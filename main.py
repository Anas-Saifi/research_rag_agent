from nodes import graph


if __name__ == "__main__":
    res = graph.invoke({"query": "give recent findings on modern llm flows, also cite the sources"})
    print(res["llm_response"])