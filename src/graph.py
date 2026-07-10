from src.state import State
from src.agent import (cluster_node, evaluate_node, reflection_node,)
from langgraph.graph import StateGraph, START, END

def should_continue(state: State):
    return state["decision"]

builder = StateGraph(State)

builder.add_node("cluster", cluster_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("reflection", reflection_node)

builder.add_edge(START, "cluster")

builder.add_edge("cluster", "evaluate")

builder.add_edge("evaluate", "reflection")

builder.add_conditional_edges("reflection", should_continue, {"continue": "cluster", "stop": END,},)

graph = builder.compile()