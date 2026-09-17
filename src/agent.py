
import time

from src.state import State
from src.clustering import run_clustering
from src.new_evaluation import evaluate_clustering

from src.no_biology_prompt import (
    build_reflection_prompt as build_no_biology_reflection_prompt,
    build_final_selection_prompt as build_no_biology_final_selection_prompt,
)

from src.wt_biology_prompts import (
    build_reflection_prompt as build_biology_reflection_prompt,
    build_final_selection_prompt as build_biology_final_selection_prompt,
)

from src.decision import (
    OptimizationDecision,
    FinalExperimentSelection,
)

from src.openAI_llm import llm
from src.experiment_logger import add_final_result
from src.timing import start_timer, stop_timer


# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------

MAX_ITERATIONS = 3
MAX_LLM_ATTEMPTS = 3


# -----------------------------------------------------
# STRUCTURED LLM OUTPUTS
# -----------------------------------------------------

structured_llm = llm.with_structured_output(
    OptimizationDecision
)

final_selection_llm = llm.with_structured_output(
    FinalExperimentSelection
)


# -----------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------

def get_config_key(parameters):
    """
    Create a unique key for a clustering configuration.

    Used to prevent the agent from evaluating
    the same parameter combination more than once.
    """

    return (
        parameters["n_neighbors"],
        parameters["resolution"],
        parameters["n_pcs"],
    )


def validate_parameters(parameters):
    """
    Validate a proposed clustering parameter configuration.

    Returns:
        The validated parameter dictionary.

    Raises:
        ValueError: If required parameters are missing
        or parameter values are outside the allowed ranges.
    """

    required_parameters = {
        "n_neighbors",
        "resolution",
        "n_pcs",
    }

    # --------------------------------------------------------
    # Check required fields
    # --------------------------------------------------------

    if not required_parameters.issubset(
        parameters.keys()
    ):

        missing_parameters = (
            required_parameters - parameters.keys()
        )

        raise ValueError(
            "The LLM returned an incomplete parameter "
            f"configuration. Missing: {missing_parameters}"
        )

    # --------------------------------------------------------
    # Validate n_neighbors
    # --------------------------------------------------------

    if not (
        10 <= parameters["n_neighbors"] <= 50
    ):

        raise ValueError(
            "n_neighbors must be between 10 and 50."
        )

    # --------------------------------------------------------
    # Validate n_pcs
    # --------------------------------------------------------

    if not (
        10 <= parameters["n_pcs"] <= 50
    ):

        raise ValueError(
            "n_pcs must be between 10 and 50."
        )

    # --------------------------------------------------------
    # Validate resolution
    # --------------------------------------------------------

    if not (
        0.1 <= parameters["resolution"] <= 2.0
    ):

        raise ValueError(
            "resolution must be between 0.1 and 2.0."
        )

    return parameters


def build_reflection_prompt(state: State):
    """
    Build the appropriate reflection prompt depending
    on whether biological evidence is available.
    """

    parameters = state["parameters"]
    metrics = state["metrics"]

    history_for_llm = [
        experiment
        for experiment in state["history"]
        if experiment["iteration"] < state["iteration"]]

    evaluated_configs = (state["evaluated_configs"].copy())

    if state["condition"] == "biology":

        return build_biology_reflection_prompt(
            history_for_llm,
            parameters,
            metrics,
            evaluated_configs,
            MAX_ITERATIONS,)

    return build_no_biology_reflection_prompt(
        history_for_llm,
        parameters,
        metrics,
        evaluated_configs,
        MAX_ITERATIONS,
    )


def build_final_selection_prompt(state: State):
    """
    Build the appropriate final-selection prompt depending
    on whether biological evidence is available.
    """

    history = state["history"]

    if state["condition"] == "biology":

        return build_biology_final_selection_prompt(history)

    return build_no_biology_final_selection_prompt(history)


# -----------------------------------------------------
# CLUSTERING NODE
# -----------------------------------------------------

def cluster_node(state: State):
    """
    Run Leiden clustering using the current parameters
    and measure clustering runtime.
    """

    start_time = time.perf_counter()

    adata = state["adata"]
    parameters = state["parameters"]

    adata = run_clustering(
        adata,
        n_neighbors=parameters["n_neighbors"],
        resolution=parameters["resolution"],
        n_pcs=parameters["n_pcs"],)

    clustering_time = (time.perf_counter() - start_time)

    return {
        "adata": adata,
        "clustering_time": clustering_time,}


# -----------------------------------------------------
# EVALUATION NODE
# -----------------------------------------------------

def evaluate_node(state: State):
    """
    Evaluate the current clustering and store the
    resulting experiment in the history and experiment log.
    """

    adata = state["adata"]

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    start_time = start_timer()

    include_markers = (state["condition"] == "biology")

    metrics = evaluate_clustering(
        adata,
        include_markers=include_markers,)

    evaluation_time = stop_timer(start_time)

    # --------------------------------------------------------
    # Create experiment history entry
    # --------------------------------------------------------

    experiment = {
        "iteration": state["iteration"],
        "parameters": state["parameters"].copy(),
        "metrics": metrics,}

    history = state["history"].copy()
    history.append(experiment)

    # --------------------------------------------------------
    # Create detailed iteration log
    # --------------------------------------------------------

    iteration_log = {
        "iteration": state["iteration"],

        "parameters": (state["parameters"].copy()),

        "metrics": {
            "silhouette_score": (metrics["silhouette_score"]),
            "davies_bouldin_score": (metrics["davies_bouldin_score"]),
            "wc_dispersion_score": (metrics["wc_dispersion_score"]),
            "banfield_raftery_score": (metrics["banfield_raftery_score"]),
        },

        "cluster_structure": {
            "n_clusters": metrics["n_clusters"],
            "cluster_sizes": (metrics["cluster_sizes"]),
            "largest_cluster_size": (metrics["largest_cluster_size"]),
            "largest_cluster_fraction": (metrics["largest_cluster_fraction"]),
            "number_of_small_clusters": (metrics["number_of_small_clusters"]),
            "small_clusters": (metrics["small_clusters"]),
        },

        "clustering_time_seconds": (state["clustering_time"]),

        "evaluation_time_seconds": (evaluation_time),
    }

    # --------------------------------------------------------
    # Add biological evidence when applicable
    # --------------------------------------------------------

    if state["condition"] == "biology":

        iteration_log["marker_genes"] = (metrics["marker_genes"])

    # --------------------------------------------------------
    # Update experiment log
    # --------------------------------------------------------

    experiment_log = (state["experiment_log"].copy())

    experiment_log["iterations"].append(iteration_log)

    # --------------------------------------------------------
    # Track evaluated configuration
    # --------------------------------------------------------

    config_key = get_config_key(state["parameters"])

    evaluated_configs = (state["evaluated_configs"].copy())

    if config_key not in evaluated_configs:

        evaluated_configs.append(config_key)

    # --------------------------------------------------------
    # Return updated state
    # --------------------------------------------------------

    return {
        "metrics": metrics,
        "history": history,
        "evaluated_configs": evaluated_configs,
        "experiment_log": experiment_log,
        "evaluation_time": evaluation_time,}


# -----------------------------------------------------
# REFLECTION NODE
# -----------------------------------------------------

def reflection_node(state: State):
    """
    Use an LLM to decide whether further parameter
    exploration is justified.

    If the LLM chooses "continue", a valid and previously
    unevaluated parameter configuration is returned.

    If the LLM chooses "stop", the optimization terminates
    and the workflow proceeds to final experiment selection.
    """

    # -----------------------------------------------------
    # 1. CHECK MAXIMUM EXPERIMENT BUDGET
    # -----------------------------------------------------

    completed_experiments = len(state["history"])

    if completed_experiments >= MAX_ITERATIONS:

        print("\n==============================")
        print("MAXIMUM EXPERIMENTS REACHED")
        print("==============================")

        print(
            f"Maximum number of experiments "
            f"({MAX_ITERATIONS}) reached.")

        print(
            "\nParameter exploration has stopped. "
            "The completed experiment history will "
            "be used for final configuration selection.")

        experiment_log = (state["experiment_log"].copy())

        experiment_log["termination_reason"] = "maximum_experiments_reached"

        return {
            "parameters": state["parameters"],
            "decision": "stop",
            "reason": (
                f"Maximum number of experiments "
                f"({MAX_ITERATIONS}) reached."),
            "experiment_log": experiment_log,
            "iteration": state["iteration"],
            "llm_decision_time": 0.0,}

    # -----------------------------------------------------
    # 2. BUILD REFLECTION PROMPT
    # -----------------------------------------------------

    prompt = build_reflection_prompt(state)

    # -----------------------------------------------------
    # 3. DISPLAY CURRENT EVALUATION
    # -----------------------------------------------------

    metrics = state["metrics"]

    print("\n==============================")
    print("CURRENT EVALUATION")
    print("==============================")

    print(
        f"Clusters: "
        f"{metrics['n_clusters']}")

    print(
        f"Silhouette: "
        f"{metrics['silhouette_score']:.4f}")

    print(
        f"DBI: "
        f"{metrics['davies_bouldin_score']:.4f}")

    print(
        f"WC-dispersion: "
        f"{metrics['wc_dispersion_score']:.4f}")

    print(
        f"Banfield-Raftery: "
        f"{metrics['banfield_raftery_score']:.4f}")

    if state["iteration"] == 0:

        print("\n==============================")
        print("LLM PROMPT")
        print("==============================")

        print(prompt)

    # -----------------------------------------------------
    # 4. REQUEST LLM DECISION
    # -----------------------------------------------------

    llm_decision_time = 0.0

    for attempt in range(
        MAX_LLM_ATTEMPTS):

        try:

            start_time = start_timer()

            decision = (structured_llm.invoke(prompt))

            llm_decision_time += (stop_timer(start_time))

            # ------------------------------------------------
            # Basic response validation
            # ------------------------------------------------

            if decision is None:

                raise ValueError("The LLM returned no decision.")

            decision_type = (
                str(decision.decision)
                .lower()
                .strip())

            if decision_type not in {
                "continue",
                "stop",}:

                raise ValueError(
                    "Unsupported decision: "
                    f"{decision.decision}")

            # -----------------------------------------------------
            # 5. HANDLE STOP DECISION
            # -----------------------------------------------------

            if decision_type == "stop":

                print("\n==============================")
                print("STOP DECISION")
                print("==============================")

                print(
                    "The LLM decided that further "
                    "parameter exploration is no "
                    "longer necessary.")

                print(
                    f"Reason: {decision.reason}")

                experiment_log = (
                    state["experiment_log"].copy())

                experiment_log[
                    "iterations"
                ][-1]["decision"] = {

                    "decision": "stop",

                    "reason": (
                        decision.reason
                    ),

                    "llm_decision_time_seconds": (
                        llm_decision_time
                    ),
                }

                experiment_log[
                    "termination_reason"
                ] = "llm_stop_decision"

                return {
                    "parameters": (state["parameters"]),

                    "decision": "stop",

                    "reason": (decision.reason),

                    "experiment_log": (experiment_log),

                    "iteration": (state["iteration"]),

                    "llm_decision_time": (llm_decision_time),}

            # -----------------------------------------------------
            # 6. HANDLE CONTINUE DECISION
            # -----------------------------------------------------

            # parameters is now optional in decision.py.
            # Therefore, explicitly check that the LLM
            # actually supplied parameters for CONTINUE.

            if decision.parameters is None:

                raise ValueError(
                    "The LLM returned 'continue' "
                    "but did not provide parameters.")

            candidate_parameters = (decision.parameters.model_dump())

            # ------------------------------------------------
            # Validate parameter values
            # ------------------------------------------------

            candidate_parameters = (validate_parameters(candidate_parameters))

            # ------------------------------------------------
            # Check duplicate configuration
            # ------------------------------------------------

            candidate_key = (get_config_key(candidate_parameters))

            evaluated_configs = (state["evaluated_configs"].copy())

            if candidate_key in evaluated_configs:

                raise ValueError(
                    "The proposed configuration "
                    f"{candidate_key} has already "
                    "been evaluated.")

            # -----------------------------------------------------
            # 7. VALID CONTINUE DECISION
            # -----------------------------------------------------

            print("\n==============================")
            print("CONTINUE DECISION")
            print("==============================")

            print(
                f"Next parameters: "
                f"{candidate_parameters}")

            print(
                f"LLM decision time: "
                f"{llm_decision_time:.2f} seconds")

            # ------------------------------------------------
            # Record decision in experiment log
            # ------------------------------------------------

            experiment_log = (state["experiment_log"].copy())

            experiment_log["iterations"][-1]["decision"] = {
                
                "decision": "continue",
                
                "parameters": (candidate_parameters),

                "reason": (decision.reason),

                "llm_decision_time_seconds": (llm_decision_time),
            }

            # ------------------------------------------------
            # Return updated state
            # ------------------------------------------------

            return {
                "parameters": (candidate_parameters),

                "decision": "continue",

                "reason": (decision.reason),

                "experiment_log": (experiment_log),

                "iteration": (state["iteration"] + 1),

                "llm_decision_time": (llm_decision_time),
            }

        # -----------------------------------------------------
        # 8. HANDLE INVALID LLM RESPONSE
        # -----------------------------------------------------

        except Exception as e:

            print("\n==============================")
            print(
                "LLM RESPONSE VALIDATION FAILED"
            )
            print("==============================")

            print(
                f"Attempt {attempt + 1} "
                f"of {MAX_LLM_ATTEMPTS} failed.")

            print(
                f"Error: {e}")

            # ------------------------------------------------
            # Retry if attempts remain
            # ------------------------------------------------

            if attempt < (
                MAX_LLM_ATTEMPTS - 1):

                print(
                    "Requesting another valid "
                    "optimization decision...")

                prompt += """

IMPORTANT:

Your previous response could not be accepted.

Return a valid structured optimization decision.

The decision must be exactly:

"continue"

or:

"stop"

If the decision is "continue":

- provide exactly one parameter configuration;
- n_neighbors must be between 10 and 50;
- n_pcs must be between 10 and 50;
- resolution must be between 0.1 and 2.0;
- do not propose a configuration that has
  already been evaluated.

If the decision is "stop":

- provide a concise reason based only on
  the available clustering evidence.

Already evaluated configurations are:

""" + str(
                    state[
                        "evaluated_configs"
                    ]
                )

                continue

            # ------------------------------------------------
            # All attempts failed
            # ------------------------------------------------

            raise RuntimeError(
                "The LLM failed to produce a valid "
                "optimization decision after "
                f"{MAX_LLM_ATTEMPTS} attempts."
            ) from e


# -----------------------------------------------------
# FINAL EXPERIMENT SELECTION NODE
# -----------------------------------------------------

def final_selection_node(state: State):
    """
    Use an LLM to select the preferred experiment
    from the completed optimization history.
    """

    history = state["history"]

    # -----------------------------------------------------
    # 1. CHECK FOR COMPLETED EXPERIMENTS
    # -----------------------------------------------------

    if not history:

        print("\n==============================")
        print("FINAL EXPERIMENT SELECTION")
        print("==============================")

        print(
            "No completed experiments are available "
            "for final selection.")

        reason = (
            "No completed experiments were available "
            "for final selection.")

        experiment_log = (
            state["experiment_log"].copy())

        experiment_log = add_final_result(
            experiment_log,
            None,
            final_selection_reason=reason,
            final_selection_time=0.0,)

        return {
            "selected_iteration": None,
            "best_experiment": None,
            "final_selection_reason": reason,
            "final_selection_time": 0.0,
            "experiment_log": experiment_log,}

    # -----------------------------------------------------
    # 2. BUILD FINAL SELECTION PROMPT
    # -----------------------------------------------------

    prompt = build_final_selection_prompt(state)

    # -----------------------------------------------------
    # 3. DISPLAY SELECTION INFORMATION
    # -----------------------------------------------------

    print("\n==============================")
    print("FINAL EXPERIMENT SELECTION")
    print("==============================")

    print(
        f"Comparing {len(history)} completed "
        "experiments.")

    # -----------------------------------------------------
    # 4. REQUEST FINAL LLM SELECTION
    # -----------------------------------------------------

    start_time = start_timer()

    final_decision = (final_selection_llm.invoke(prompt))

    final_selection_time = (stop_timer(start_time))

    # -----------------------------------------------------
    # 5. VALIDATE FINAL LLM RESPONSE
    # -----------------------------------------------------

    if final_decision is None:

        raise RuntimeError(
            "The final selection LLM returned "
            "no decision.")

    selected_iteration = (final_decision.selected_iteration)

    available_iterations = {experiment["iteration"] for experiment in history}

    if (
        selected_iteration
        not in available_iterations
    ):

        print("\n==============================")
        print("INVALID FINAL SELECTION")
        print("==============================")

        print(
            f"LLM selected iteration "
            f"{selected_iteration}, but this "
            "iteration does not exist in the "
            "completed experiment history."
        )

        invalid_reason = (
            "The LLM selected an iteration that "
            "does not exist in the completed "
            "experiment history."
        )

        experiment_log = (
            state["experiment_log"].copy()
        )

        experiment_log = add_final_result(
            experiment_log,
            None,
            final_selection_reason=(
                invalid_reason
            ),
            final_selection_time=(
                final_selection_time
            ),
        )

        return {
            "selected_iteration": None,
            "best_experiment": None,
            "final_selection_reason": (
                invalid_reason
            ),
            "final_selection_time": (
                final_selection_time
            ),
            "experiment_log": (
                experiment_log
            ),
        }

    # -----------------------------------------------------
    # 6. RETRIEVE SELECTED EXPERIMENT
    # -----------------------------------------------------

    best_experiment = next(
        experiment
        for experiment in history
        if experiment["iteration"]
        == selected_iteration
    )

    # -----------------------------------------------------
    # 7. DISPLAY FINAL SELECTION
    # -----------------------------------------------------

    print("\n==============================")
    print("FINAL LLM SELECTION")
    print("==============================")

    print(
        f"Selected iteration: "
        f"{selected_iteration}"
    )

    print(
        f"Selected parameters: "
        f"{best_experiment['parameters']}"
    )

    print(
        f"Reason: "
        f"{final_decision.reason}"
    )

    print(
        f"Final selection time: "
        f"{final_selection_time:.2f} seconds"
    )

    # -----------------------------------------------------
    # 8. RECORD FINAL SELECTION
    # -----------------------------------------------------

    experiment_log = (
        state["experiment_log"].copy()
    )

    experiment_log = add_final_result(
        experiment_log,
        best_experiment,
        final_selection_reason=(
            final_decision.reason
        ),
        final_selection_time=(
            final_selection_time
        ),
    )

    # -----------------------------------------------------
    # 9. RETURN FINAL RESULT
    # -----------------------------------------------------

    return {
        "selected_iteration": (
            selected_iteration
        ),

        "best_experiment": (
            best_experiment
        ),

        "final_selection_reason": (
            final_decision.reason
        ),

        "final_selection_time": (
            final_selection_time
        ),

        "experiment_log": (
            experiment_log
        ),
    }

