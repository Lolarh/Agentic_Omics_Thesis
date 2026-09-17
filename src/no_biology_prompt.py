def build_reflection_prompt(
    history,
    parameters,
    metrics,
    evaluated_configs,
    max_iterations,
):
    return f"""
You are an expert in single-cell RNA-seq clustering optimization.

Your objective is to explore clustering parameters and identify
parameter configurations that provide strong overall clustering
evidence while avoiding unnecessary over-fragmentation or
under-clustering.

The optimization should balance potential improvement against the
cost of additional experiments. Do not continue exploring merely
for the sake of using the available experiment budget.

NO-BIOLOGY EXPERIMENTAL CONDITION:

This is the no-biology experimental condition.

* Do not use marker genes, cell-type annotations, biological labels,
  known cell identities, or biological interpretations.
* Base all optimization decisions exclusively on intrinsic clustering
  metrics and cluster-structure information provided in the prompt.
* Do not infer or assume biological cell identities.
* Biological plausibility must not be used as a criterion for parameter
  selection, comparison, or stopping decisions.

CURRENT EXPERIMENT:

Current clustering parameters:
{parameters}

Current evaluation metrics:
{metrics}

Previous optimization history:
{history}

Already evaluated parameter configurations:
{evaluated_configs}

EVALUATION PRINCIPLES:

* Use Silhouette Score as evidence of cluster separation and cohesion,
  where higher values are generally favorable.

* Use Davies-Bouldin Index as additional evidence of cluster quality,
  where lower values are generally favorable.

* Use WC-dispersion as evidence of within-cluster dispersion. Lower
  values indicate more compact clusters, but do not optimize
  WC-dispersion in isolation because cluster subdivision can reduce
  within-cluster dispersion.

* Use Banfield-Raftery as an additional intrinsic clustering criterion
  related to within-cluster compactness and partition quality. Lower
  values may be favorable according to this criterion, but do not
  optimize it in isolation because it may also be influenced by the
  number and partitioning of clusters.

* Evaluate Silhouette Score, Davies-Bouldin Index, WC-dispersion,
  Banfield-Raftery, number of clusters, and cluster-size distribution
  together.

* Do not automatically consider a configuration better simply because
  one metric improves. Look for agreement and disagreement among the
  available evidence and consider relevant trade-offs.

* Compare configurations across the optimization history. Consider
  whether parameter changes produce meaningful overall improvements
  rather than interpreting individual metric values in isolation.

CLUSTER STRUCTURE AND OVER-FRAGMENTATION:

* Consider the number of clusters and the cluster-size distribution
  when assessing possible under-clustering or over-clustering.

* Higher resolution may increase cluster granularity, but excessive
  resolution can lead to unnecessary fragmentation.

* Lower resolution may merge potentially distinct populations.
  Do not assume that fewer clusters are automatically better.

* Use cluster sizes, largest_cluster_fraction, and
  number_of_small_clusters as diagnostic evidence rather than
  automatic rejection criteria.

* Extremely small clusters may indicate over-fragmentation, but should
  be interpreted alongside the other available structural evidence.

* A highly dominant cluster may indicate under-clustering, but
  dominance alone is not sufficient evidence to reject a solution.

PARAMETER REASONING:

* Consider the combined effects of n_neighbors, n_pcs, and resolution.

* Do not assume that increasing or decreasing any individual parameter
  will always improve clustering.

* Lower n_neighbors may emphasize local structure, while higher values
  may produce a smoother neighborhood graph.

* Increasing n_pcs may capture additional variation but may also
  introduce less informative variation.

* Consider the optimization history when choosing the next parameter
  configuration.

* Prefer parameter changes that provide an informative test of the
  current clustering evidence.

* Changing one parameter while holding others constant can be useful
  when it helps isolate the effect of that parameter.

* Avoid arbitrary parameter changes that are not supported by the
  current metrics or optimization history.

PARAMETER CONSTRAINTS:

* n_neighbors must be between 10 and 50.

* n_pcs must be between 10 and 50 because only 50 PCA components are
  available.

* resolution must be between 0.1 and 2.0.

* The resolution range of 0.1 to 2.0 is the experimental search space
  defined for this study. It is not an inherent limitation of the
  Leiden algorithm.

* Do not assume that higher resolution is better simply because it
  produces more clusters.

* Do not propose parameter values outside these ranges.

* Do not propose a parameter configuration that has already been
  evaluated.

STABILITY:

* Do not claim that a clustering solution is formally stable or
  reproducible unless stability has been explicitly assessed through
  repeated clustering under different random seeds or data
  perturbations.

* Consistency across different parameter configurations is not itself
  a formal stability assessment.

EXPLORATION BUDGET:

* The optimization has a maximum exploration budget of
  {max_iterations} completed clustering evaluations.

* This maximum is a safety limit, not a requirement to use all
  available experiments.

* Iteration 0 represents the initial clustering configuration.

* Subsequent iterations represent agentic parameter updates based on
  the evidence from previous experiments.

* The optimization may terminate before the maximum budget is reached
  if the available evidence suggests that further parameter
  exploration is unlikely to provide meaningful improvement.

* Do not assume that reaching the maximum number of experiments is
  necessary for a good result.

* At each reflection step, consider whether another parameter
  configuration is likely to provide useful additional information.

* Do not propose a configuration that has already been evaluated.

* The purpose of the exploration budget is to provide sufficient
  opportunity to explore the parameter space while preventing
  unnecessarily long optimization runs.

DECISION:

* Return "continue" when another parameter configuration should be
  evaluated and the available evidence supports further exploration.

* Return "stop" when the current experiment history provides
  sufficient evidence and further parameter exploration is unlikely
  to provide meaningful additional information.

* When deciding whether to stop, consider whether recent parameter
  changes have produced meaningful improvement and whether additional
  exploration is likely to provide useful new information.

* Do not stop simply because the most recent experiment is not better
  than the previous experiment.

* Likewise, do not continue simply because unused experiment budget
  remains.

* Do not return "stop" solely because one metric is unfavorable.

* When continuing, propose exactly one new, unevaluated parameter
  configuration.

* When stopping, do not propose a new configuration.

* Briefly explain the evidence supporting the decision.


OUTPUT REQUIREMENTS:

Return a structured optimization decision containing:

1. decision:
   "continue" if further exploration is justified, or
   "stop" if further exploration is no longer justified.

2. reason:
   A concise explanation of what the experiments indicate and why
   the optimization should continue or stop.

3. parameters:
   If decision is "continue", provide a valid new parameter
   configuration that has not already been evaluated.

   If decision is "stop", the parameters are not used.
"""

def build_final_selection_prompt(history):
    """
    Build the prompt used by the LLM to select the preferred
    experiment from the completed no-biology experiment history.
    """

    return f"""
You are an expert in clustering evaluation and single-cell RNA-seq
data analysis.

Your task is to select the preferred clustering experiment from the
completed experiment history.

NO-BIOLOGY EXPERIMENTAL CONDITION:

This is the no-biology experimental condition.

* Do not use marker genes, cell-type annotations, biological labels,
  known cell identities, or biological interpretations.
* Do not infer biological identities from the clustering results.
* Biological plausibility must not be used as a criterion for selecting
  the preferred configuration.
* Base the final selection exclusively on intrinsic clustering metrics
  and cluster-structure information contained in the experiment
  history.

COMPLETED EXPERIMENT HISTORY:

{history}

FINAL SELECTION PRINCIPLES:

* Select only one experiment that already exists in the completed
  experiment history.

* Your selected_iteration must exactly match the iteration number of
  one completed experiment.

* Do not propose new clustering parameters.

* Evaluate the experiments comparatively rather than judging any
  metric in isolation.

* Consider Silhouette Score as evidence of cluster separation and
  cohesion, where higher values are generally favorable.

* Consider Davies-Bouldin Index as additional evidence of clustering
  quality, where lower values are generally favorable.

* Consider WC-dispersion as evidence of within-cluster compactness.
  Lower values may indicate more compact clusters, but do not select
  an experiment solely because it has the lowest WC-dispersion because
  partitioning into more clusters can reduce within-cluster dispersion.

* Consider Banfield-Raftery as an additional intrinsic clustering
  criterion. Interpret it alongside the other metrics rather than
  selecting an experiment solely because of this value.

* Consider the number of clusters and cluster-size distribution as
  contextual structural evidence.

* Extremely small clusters may indicate over-fragmentation, but should
  be interpreted alongside the other available structural evidence.

* A highly dominant cluster may indicate under-clustering, but this
  alone is not sufficient evidence to reject an experiment.

* Look for overall agreement among the available intrinsic evidence.

* When metrics disagree, explain the relevant trade-off and select the
  experiment that provides the strongest overall intrinsic clustering
  evidence rather than optimizing a single metric.

FINAL DECISION:

* Select exactly one completed experiment.

* Return the iteration number of the selected experiment.

* Briefly explain why it provides the strongest overall clustering
  evidence compared with the alternatives.

* Do not use biological plausibility or inferred cell identities in
  the explanation.
"""