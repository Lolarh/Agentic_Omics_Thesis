from src.state import State
from src.clustering import run_clustering
from src.evaluation import evaluate_clustering, get_best_experiment
from src.decision import OptimizationDecision
from src.prompts import build_reflection_prompt
from src.openAI_llm import llm

structured_llm = llm.with_structured_output(OptimizationDecision)

MAX_ITERATIONS = 20

CONVERGENCE_PATIENCE = 3 # Stop when there have been 3 consecutive experiments without meaningful improvement.


def get_config_key(parameters):
    return (
        parameters["n_neighbors"],
        parameters["resolution"],
        parameters["n_pcs"])


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

    # Record this configuration as evaluated
    config_key = get_config_key(state["parameters"])

    evaluated_configs = state["evaluated_configs"].copy()

    if config_key not in evaluated_configs:
        evaluated_configs.append(config_key)

    return {
        "metrics": metrics,
        "history": history,
        "evaluated_configs": evaluated_configs,}
""" Storing metrics here because the next node (the LLM) will need to reason about them. For instance, if Silhouette is low, it could increase the number of neighbours"""

#### The decision-making component of the agent. 
# Looks at the results of the previous experiment, decides whether the clustering has improved, and determines what parameter should be tested next or whether the optimization should stop.

def reflection_node(state: State):
    """
    Use an LLM to reflect on the current clustering
    and propose the next parameter configuration.
    """

    # --------------------------------------------------
    # 1. Safety guardrail
    # --------------------------------------------------

    if state["iteration"] >= MAX_ITERATIONS:

        best_experiment = get_best_experiment(
            state["history"]
        )

        return {
            "parameters": (
                best_experiment["parameters"]
                if best_experiment
                else state["parameters"]
            ),
            "decision": "stop",
            "reason": (
                f"Maximum iteration limit "
                f"({MAX_ITERATIONS}) reached."),
            "iteration": state["iteration"],}

    # --------------------------------------------------
    # 2. Prepare information for the LLM
    # --------------------------------------------------

    parameters = state["parameters"]
    metrics = state["metrics"]

    history_for_llm = [experiment for experiment in state["history"] if experiment["iteration"] < state["iteration"]]

    evaluated_configs = state["evaluated_configs"]

    prompt = build_reflection_prompt(
        history_for_llm,
        parameters,
        metrics,
        evaluated_configs,)

    # --------------------------------------------------
    # 3. Display current evaluation
    # --------------------------------------------------

    print("\n==============================")
    print("CURRENT EVALUATION")
    print("==============================")

    print(f"Clusters: {metrics['n_clusters']}")
    print(
        f"Silhouette: "
        f"{metrics['silhouette_score']:.4f}")
    print(
        f"DBI: "
        f"{metrics['davies_bouldin_score']:.4f}")
    print(
        f"WC-dispersion: "
        f"{metrics['wc_dispersion_score']:.4f}")

    if state["iteration"] == 0:
        print("\n==============================")
        print("LLM PROMPT")
        print("==============================")
        print(prompt)

    # --------------------------------------------------
    # 4. Ask LLM for next decision
    # --------------------------------------------------

    decision = None

    for attempt in range(3):

        decision = structured_llm.invoke(prompt)

        # Normalize decision value
        decision_type = decision.decision.lower().strip()

        # --------------------------------------------------
        # If LLM explicitly says STOP
        # --------------------------------------------------

        if decision_type == "stop":

            best_experiment = get_best_experiment(state["history"])

            print("\n==============================")
            print("LLM DECISION")
            print("==============================")

            print("Decision: stop")
            print(f"Reason: {decision.reason}")

            if best_experiment:
                print("\n==============================")
                print("BEST EXPERIMENT")
                print("==============================")

                print(
                    f"Best iteration: "
                    f"{best_experiment['iteration']}")

                print(
                    f"Best parameters: "
                    f"{best_experiment['parameters']}")

                print(
                    f"Best Silhouette: "
                    f"{best_experiment['metrics']['silhouette_score']:.4f}")

                print(
                    f"Best DBI: "
                    f"{best_experiment['metrics']['davies_bouldin_score']:.4f}")

            return {
                "parameters": (
                    best_experiment["parameters"]
                    if best_experiment
                    else state["parameters"]),
                "decision": "stop",
                "reason": decision.reason,
                "iteration": state["iteration"],}

        # --------------------------------------------------
        # If LLM says CONTINUE, validate configuration
        # --------------------------------------------------

        if decision_type == "continue":

            candidate_parameters = (decision.parameters.model_dump())

            candidate_key = get_config_key(candidate_parameters)

            if candidate_key not in evaluated_configs:
                break

            print("\n==============================")
            print("DUPLICATE CONFIGURATION")
            print("==============================")

            print(f"Proposed: {candidate_key}")

            print("Configuration already evaluated.")

            print("Requesting another configuration...")

            prompt += f"""
IMPORTANT:
The configuration {candidate_key}
has already been evaluated.

You must propose a different,
unevaluated configuration.

Already evaluated configurations:
{evaluated_configs}
"""

            continue

        # --------------------------------------------------
        # Unknown decision
        # --------------------------------------------------

        print("\n==============================")
        print("INVALID LLM DECISION")
        print("==============================")

        print(
            f"Received unsupported decision: "
            f"{decision.decision}")

        return {
            "parameters": state["parameters"],
            "decision": "stop",
            "reason": (
                f"Unsupported LLM decision: "
                f"{decision.decision}"),
            "iteration": state["iteration"],}

    # --------------------------------------------------
    # 5. Check whether a valid new configuration exists
    # --------------------------------------------------

    candidate_parameters = (decision.parameters.model_dump())

    candidate_key = get_config_key(candidate_parameters)

    if candidate_key in evaluated_configs:

        return {
            "parameters": state["parameters"],
            "decision": "stop",
            "reason": (
                "Unable to generate a new "
                "unevaluated parameter configuration "
                "after 3 attempts."),
            "iteration": state["iteration"],}

    # --------------------------------------------------
    # 6. Display CONTINUE decision
    # --------------------------------------------------

    print("\n==============================")
    print("LLM DECISION")
    print("==============================")

    print(f"Decision: {decision.decision}")

    print(f"n_neighbors: "f"{decision.parameters.n_neighbors}")

    print(f"resolution: "f"{decision.parameters.resolution}")

    print(f"n_pcs: "f"{decision.parameters.n_pcs}")

    print(f"Reason: {decision.reason}")

    # --------------------------------------------------
    # 7. Update state and continue optimization
    # --------------------------------------------------

    return {
        "parameters": decision.parameters.model_dump(),
        "decision": "continue",
        "reason": decision.reason,
        "iteration": state["iteration"] + 1,}