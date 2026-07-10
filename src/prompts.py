def build_reflection_prompt(history, parameters, metrics):
    return f""" 
    
    You are an expert in single-cell RNA-seq clustering optimization.
    Your objective is to improve clustering quality.
    Current clustering parameters:{parameters}
    Current evaluation metrics:{metrics}
    Previous optimization history: {history}
    
    Instructions:
    - Maximize the Silhouette Score.
    - Minimize the Davies–Bouldin Index.
    - Consider previous experiments before suggesting new parameters.
    - Adjust only ONE clustering parameter at a time.
    - If no meaningful improvement is likely, return "stop".
    - Briefly explain your reasoning.
    
    Return your response using the provided schema."""
