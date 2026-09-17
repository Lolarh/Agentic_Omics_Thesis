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
cost of additional experiments. Do not continue exploring merely for
the sake of using the available experiment budget.

BIOLOGY-INFORMED EXPERIMENTAL CONDITION:

This is the biology-informed experimental condition.

* Use intrinsic clustering metrics, cluster-structure information,
  and biological evidence provided in the prompt when making
  optimization decisions.

* Biological evidence may be used to assess whether clusters appear
  biologically coherent, whether potentially distinct populations may
  have been merged, and whether small clusters may represent
  meaningful biological structure rather than unnecessary
  fragmentation.

* Do not force the clustering toward a predetermined number of
  clusters or a predetermined biological outcome.

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
  Banfield-Raftery, number of clusters, cluster-size distribution,
  and biological evidence together.

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

* Extremely small clusters may indicate over-fragmentation, but rare
  biological populations may also form meaningful small clusters.

* A highly dominant cluster may indicate under-clustering, but
  dominance alone is not sufficient evidence to reject a solution.

BIOLOGICAL EVIDENCE:

The evaluation may include top-ranked marker genes for each cluster:

{metrics.get("marker_genes", {})}

Use these marker genes as biological evidence when assessing the
clustering configuration.

Consider:

* whether marker genes within a cluster suggest a coherent
  cellular population;

* whether different clusters show distinct marker patterns;

* whether a configuration may be merging biologically distinct
  populations;

* whether additional clusters appear to represent plausible
  biological structure or simply fragmentation.
  
Interpret groups of marker genes together rather than relying on a
single marker gene.

Do not treat marker genes as definitive cell-type labels.

Do not force the clustering toward a predetermined number of
clusters. Biological evidence should be considered together with
the mathematical and structural evidence.

PBMC BIOLOGICAL CONTEXT:

* PBMC3K contains several broad immune-cell populations.
  Canonical marker-gene patterns can therefore provide useful context
  when assessing whether clusters represent distinct biological
  populations.

* This context is a soft biological prior and should not be treated
  as a fixed target for the number of clusters.

* Use the expected broad cellular diversity of PBMC3K as contextual
  evidence when assessing clustering granularity. If a configuration
  produces substantially more clusters than the expected broad
  population structure, assess whether the additional clusters are
  supported by distinct marker-gene patterns or instead indicate
  unnecessary splitting of similar populations.

* Likewise, if substantially fewer clusters are produced, assess
  whether biologically distinct populations may have been merged.

* The expected number of broad populations should guide interpretation,
  not act as a fixed target for the clustering result.

PARAMETER REASONING:

* Consider the combined effects of n_neighbors, n_pcs, and resolution.

* Do not assume that increasing or decreasing any individual
  parameter will always improve clustering.

* Lower n_neighbors may emphasize local structure, while higher
  values may produce a smoother neighborhood graph.

* Increasing n_pcs may capture additional variation but may also
  introduce less informative variation.

* Consider the optimization history when choosing the next parameter
  configuration.

* Prefer parameter changes that provide an informative test of the
  current clustering evidence.

* Changing one parameter while holding others constant can be useful
  when it helps isolate the effect of that parameter.

* Avoid arbitrary parameter changes that are not supported by the
  current metrics, biological evidence, or optimization history.

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

* The optimization has a maximum exploration budget defined by the
  agent.

* The maximum number of completed clustering evaluations is provided
  by the agent's experiment configuration.

* Iteration 0 represents the initial clustering configuration.

* Subsequent iterations represent agentic parameter updates based on
  the evidence from previous experiments.

* The maximum exploration budget is a limit, not a requirement to use
  all available experiments.

* The optimization may terminate before the maximum budget is reached
  if the available evidence suggests that further parameter
  exploration is unlikely to provide meaningful improvement.

* Do not assume that reaching the maximum number of experiments is
  necessary for a good result.

* At each reflection step, consider whether another parameter
  configuration is likely to provide useful additional information.

* Do not propose a configuration that has already been evaluated.

* The purpose of the exploration budget is to provide the agent with
  sufficient opportunity to explore the parameter space while
  preventing unnecessarily long optimization runs.

DECISION:

* Decide whether the optimization should continue or stop based on
  the evidence available in the completed experiment history.

* Return "continue" when another unevaluated parameter configuration
  is likely to provide useful information or potentially improve the
  clustering solution.

* When deciding whether to stop, consider whether recent parameter
  changes have produced meaningful improvement and whether additional
  exploration is likely to provide useful new information.

* Do not stop simply because the most recent experiment is not better
  than the previous experiment.

* Likewise, do not continue simply because unused experiment budget
  remains.

* Return "stop" when the current experiment history provides sufficient
  evidence and further exploration is unlikely to provide a meaningful
  improvement.

* Do not stop solely because one metric has failed to improve in a
  single iteration.

* Consider the overall pattern across the optimization history,
  including intrinsic metrics, cluster structure, biological evidence,
  and the changes produced by previous parameter configurations.

* If the exploration budget has been reached, return "stop".

* If the exploration budget has not been reached, "stop" is still
  allowed when the evidence supports termination.

* When selecting "continue", propose exactly one new, unevaluated
  parameter configuration.

* When selecting "continue", briefly explain why the proposed
  parameter change is informative given the current intrinsic metrics,
  cluster structure, biological evidence, and previous optimization
  history.

* When selecting "stop", briefly explain why the completed experiments
  provide sufficient evidence to terminate the optimization.

OUTPUT REQUIREMENTS:

Return a structured optimization decision containing:

1. decision:
   "continue" if further exploration is likely to provide useful
   information or improve the clustering solution.

   "stop" if the current evidence is sufficient or the maximum
   exploration budget has been reached.

2. reason:
   A concise explanation of what the completed experiments indicate
   and why the optimization should continue or stop.

3. parameters:
   If decision is "continue", provide a valid new parameter
   configuration that has not already been evaluated.

   If decision is "stop", the parameter values are not used for
   further clustering.

Do not treat biological evidence as an absolute rule.

Do not invent biological explanations that are unsupported by the
provided marker-gene evidence.
"""


def build_final_selection_prompt(history):
    """
    Build the prompt used by the LLM to select the preferred
    experiment from the completed biology-informed experiment history.
    """

    return f"""
You are an expert in clustering evaluation and single-cell RNA-seq
data analysis.

Your task is to select the preferred clustering experiment from the
completed experiment history.

BIOLOGY-INFORMED EXPERIMENTAL CONDITION:

This is the biology-informed experimental condition.

* Use intrinsic clustering metrics, cluster-structure information,
  and biological evidence when selecting the preferred experiment.

* Do not force the selected experiment toward a predetermined number
  of clusters or a predetermined biological outcome.

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

INTRINSIC CLUSTERING EVIDENCE:

* Consider Silhouette Score as evidence of cluster separation and
  cohesion, where higher values are generally favorable.

* Consider Davies-Bouldin Index as additional evidence of clustering
  quality, where lower values are generally favorable.

* Consider WC-dispersion as evidence of within-cluster compactness.
  Lower values may indicate more compact clusters, but do not select
  an experiment solely because it has the lowest WC-dispersion because
  partitioning into more clusters can reduce within-cluster
  dispersion.

* Consider Banfield-Raftery as an additional intrinsic clustering
  criterion. Interpret it alongside the other metrics rather than
  selecting an experiment solely because of this value.

STRUCTURAL EVIDENCE:

* Consider the number of clusters and the complete cluster-size
  distribution as contextual evidence.

* Extremely small clusters may indicate over-fragmentation, but rare
  biological populations may also form meaningful small clusters.

* A highly dominant cluster may indicate under-clustering, but this
  alone is not sufficient evidence to reject an experiment.

* Do not assume that fewer clusters are better simply because they
  produce better intrinsic metrics.

* Do not assume that more clusters are better simply because they
  represent finer partitioning.

BIOLOGICAL EVIDENCE:

* For PBMC3K, established biological knowledge may provide broad
  context regarding expected cellular diversity.

* Use the marker-gene evidence provided in the experiment history to
  assess whether clusters have biologically coherent identities.

* Consider whether marker genes support plausible PBMC populations.

* Consider whether an experiment appears to merge biologically
  distinct populations.

* Consider whether additional clusters represent meaningful
  biological structure or unnecessary fragmentation.

* Interpret groups of marker genes together rather than relying on a
  single marker gene.

* Do not treat marker genes as definitive cell-type labels.

* Do not force the selected experiment toward a predetermined number
  of clusters.

* A higher or lower number of clusters alone does not determine which
  experiment is preferable.

* Marker-gene evidence should be interpreted together with intrinsic
  and structural evidence rather than treated as an absolute rule.

OVERALL COMPARISON:

* First identify the experiment with the strongest intrinsic
  clustering evidence.

* Then identify which experiments provide the most reasonable cluster
  structure.

* Then identify which experiment provides the strongest biological
  interpretation based on the available marker-gene evidence.

* These three preferences do not have to point to the same experiment.

* If the intrinsic-metric winner differs from the biologically
  preferred experiment, explicitly explain why.

* Do not artificially force agreement between intrinsic metrics,
  cluster structure, and biological evidence.

* Select the experiment that provides the strongest overall scientific
  evidence, considering the trade-offs between these three types of
  evidence.

* Do not select an experiment solely because it has:
    * the highest Silhouette Score,
    * the lowest Davies-Bouldin Index,
    * the lowest WC-dispersion,
    * the lowest Banfield-Raftery score,
    * the fewest clusters,
    * or the most clusters.

FINAL DECISION:

* Select exactly one completed experiment.

* Return the iteration number of the selected experiment.

* In your reasoning, explicitly state:

    1. Which experiment has the strongest intrinsic metrics.

    2. Which experiment has the most reasonable cluster structure.

    3. Which experiment has the strongest biological interpretation.

    4. Which experiment you ultimately selected.

    5. If the final selection differs from the intrinsic-metric winner,
       explain the scientific trade-off that justifies the difference.

* Briefly explain why the selected experiment provides the strongest
  overall evidence compared with the alternatives.
"""

