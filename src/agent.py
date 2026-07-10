from src.state import State
from src.clustering import run_clustering
from src.evaluation import evaluate_clustering
from src.decision import OptimizationDecision
from src.prompts import build_reflection_prompt
from src.openAI_llm import llm

structured_llm = llm.with_structured_output(OptimizationDecision)

MAX_ITERATIONS = 5

def cluster_node(state: State):
    """
    Run Leiden clustering using the current parameters stored in the graph state.
    """

    adata = state["adata"]

    params = state["parameters"]

    adata = run_clustering(
        adata,
        n_neighbors=params["n_neighbors"],
        resolution=params["resolution"],
        n_pcs=params["n_pcs"],
    )

    return {
        "adata": adata,
    }

def evaluate_node(state: State):
    """
    Evaluate the current clustering and store the experiment.
    """

    adata = state["adata"]

    metrics = evaluate_clustering(adata)

    experiment = {
        "iteration": state["iteration"],
        "parameters": state["parameters"].copy(),
        "metrics": metrics,
    }

    history = state["history"].copy()
    history.append(experiment)

    return {
        "metrics": metrics,
        "history": history,
    }

""" Storing metrics here because the next node (the LLM) will need to reason about them. For instance, if Silhouette is low, it could increase the number of neighbours"""

def manual_reflection_node(state: State):
    """
    Decide whether to continue optimizing based on previous experiments.
    """

    history = state["history"]

    params = state["parameters"].copy()

    current = history[-1]

    current_score = current["metrics"]["silhouette_score"]

    decision = "stop"
    

    TARGET_SILHOUETTE = 0.15

    # First iteration
    if len(history) == 1:

        if current_score < TARGET_SILHOUETTE:
            params["n_neighbors"] += 5
            decision = "continue"

    else:

        previous = history[-2]

        previous_score = previous["metrics"]["silhouette_score"]

        if current_score > previous_score:

            # Improvement observed
            params["resolution"] += 0.1
            decision = "continue"

        else:

            # No improvement
            decision = "stop"

    return {
        "parameters": params,
        "decision": decision,
        "iteration": state["iteration"] + 1,
    }

def reflection_node(state: State):
    """
    Reflect on clustering performance using an LLM.
    """

    if state["iteration"] >= MAX_ITERATIONS:
        return {
            "parameters": state["parameters"],
            "decision": "stop",
            "reason": f"Maximum iteration limit ({MAX_ITERATIONS}) reached.",
            "iteration": state["iteration"],
        }

    history = state["history"]

    parameters = state["parameters"]

    metrics = state["metrics"]

    prompt = build_reflection_prompt(
        history,
        parameters,
        metrics,
    )

    decision = structured_llm.invoke(prompt)

    return {
        "parameters": decision.parameters.model_dump(),
        "decision": decision.decision,
        "reason": decision.reason,
        "iteration": state["iteration"] + 1,
    }
   
