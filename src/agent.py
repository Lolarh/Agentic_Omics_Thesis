from src.state import State
from src.clustering import run_clustering
from src.evaluation import evaluate_clustering
from src.decision import OptimizationDecision
from src.prompts import build_reflection_prompt
from src.openAI_llm import llm

structured_llm = llm.with_structured_output(OptimizationDecision)

MAX_ITERATIONS = 5

def cluster_node(state: State): # The node runs the Scanpy clustering pipeline using the current parameter combination
    """
    Run Leiden clustering using the current parameters stored in the graph state.
    """

    adata = state["adata"]  # retrieves the single-cell dataset from the graph's current state

    params = state["parameters"] # retrieves the current clustering parameters that the agent wants to test

    adata = run_clustering(  
        adata,
        n_neighbors=params["n_neighbors"],
        resolution=params["resolution"],
        n_pcs=params["n_pcs"],)

    return {"adata": adata,}

def evaluate_node(state: State): # measures how well the current clustering performed and records the results as one experiment in  the optimization history
    """
    Evaluate the current clustering and store the experiment.
    """

    adata = state["adata"] # retrieves the clustered dataset produced by the previous clustering step.

    metrics = evaluate_clustering(adata) # clusters are evaluated using quality metrics,

    experiment = {# creates a record of each experiment by saving the iteration number, the parameters tested, and resulting     evaluation metrics
        "iteration": state["iteration"],
        "parameters": state["parameters"].copy(),
        "metrics": metrics,}

    history = state["history"].copy() # Saves the experiment for reference
    history.append(experiment)

    return {
        "metrics": metrics,
        "history": history,}

""" Storing metrics here because the next node (the LLM) will need to reason about them. For instance, if Silhouette is low, it could increase the number of neighbours"""

#### The decision-making component of the agent. 
# Looks at the results of the previous experiment, decides whether the clustering has improved, and determines what parameter should be tested next or whether the optimization should stop.

def reflection_node(state: State): 
    """
    Reflect on clustering performance using an LLM.
    """
    if state["iteration"] >= MAX_ITERATIONS: # This retrieve everything the LLM needs to reason and decide
        return { 
            "parameters": state["parameters"],
            "decision": "stop",
            "reason": f"Maximum iteration limit ({MAX_ITERATIONS}) reached.",
            "iteration": state["iteration"],}
    

    parameters = state["parameters"]

    metrics = state["metrics"]
    
    history = [experiment for experiment in state["history"] if experiment["iteration"] < state["iteration"]]
    
    prompt = build_reflection_prompt(
    history,
    parameters,
    metrics,)
    
    print("\n==============================")
    print("CURRENT EVALUATION")
    print("==============================")
    print(f"Clusters: {metrics['n_clusters']}")
    print(f"Silhouette: {metrics['silhouette_score']:.4f}")
    print(f"DBI: {metrics['davies_bouldin_score']:.4f}")
    
    # Print the full prompt only during the first iteration
    
    if state["iteration"] == 0:
        print("\n==============================")
        print("LLM PROMPT")
        print("==============================")
        print(prompt)
    
    decision = structured_llm.invoke(prompt)
    
    print("\n==============================")
    print("LLM DECISION")
    print("==============================")
    print(f"Decision: {decision.decision}")
    print(f"n_neighbors: {decision.parameters.n_neighbors}")
    print(f"resolution: {decision.parameters.resolution}")
    print(f"n_pcs: {decision.parameters.n_pcs}")
    print(f"Reason: {decision.reason}")

    return {
        "parameters": decision.parameters.model_dump(),
        "decision": decision.decision,
        "reason": decision.reason,
        "iteration": state["iteration"] + 1,}
   
