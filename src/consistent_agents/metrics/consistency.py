from typing import Dict, Any, List, Optional, Callable
from consistent_agents.metrics.agreement_functions import llm_as_judge, contradiction, entailment, bertscore, rouge
from consistent_agents.metrics.aggregators.pairwise import pairwise
from consistent_agents.metrics.aggregators.entropy import entropy

# Mapping from string names to agreement functions
AGREEMENT_FUNCTIONS = {
    "llm_as_judge": llm_as_judge,
    "contradiction": contradiction,
    "entailment": entailment,
    "bertscore": bertscore,
    "rouge": rouge,
}

# Mapping from string names to aggregator classes
AGGREGATORS = {
    "pairwise": pairwise,
    "entropy": entropy,
}


def score(
    outputs: List[str],
    question: Optional[str] = None,
    agreement: str = "llm_as_judge",
    agreement_params: Optional[Dict[str, Any]] = None,
    aggregator: str = "pairwise",
    aggregator_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> float:
    """Calculate consistency score using agreement function + aggregator pattern.
    
    This function evaluates how consistent model outputs are across different perturbations.
    It uses an agreement function to compute similarity between outputs and an aggregator
    to combine these scores into a single metric.
    
    Args:
        outputs: List of output strings to evaluate
        question: Optional question context (passed to agreement function)
        agreement: Name of agreement function. Options: "consistency", "contradiction", "entailment", "bertscore", "rouge"
        agreement_params: Parameters for agreement function (e.g., judge_model, model_type).
        aggregator: Name of aggregator. Options: "pairwise", "entropy"
        aggregator_params: Parameters for aggregator instantiation.
        **kwargs: Additional context passed to agreement function
        
    Returns:
        Consistency score (typically 0.0 to 1.0, or entropy value)
    """
    if agreement not in AGREEMENT_FUNCTIONS:
        raise ValueError(
            f"Unknown agreement function: {agreement}. "
            f"Available options: {list(AGREEMENT_FUNCTIONS.keys())}"
        )
    
    if aggregator not in AGGREGATORS:
        raise ValueError(
            f"Unknown aggregator: {aggregator}. "
            f"Available options: {list(AGGREGATORS.keys())}"
        )
    
    agreement_fn = AGREEMENT_FUNCTIONS[agreement]
    agreement_params = agreement_params or {}
    
    aggregator_fn = AGGREGATORS[aggregator]
    
    # Create a wrapper function that calls the agreement function with the right params
    def agreement_wrapper(output1: str, output2: str, question, **kw) -> float:
        return agreement_fn(
            output1, 
            output2, 
            question,
            **agreement_params
        )
    
    # Call aggregator function directly with outputs, agreement_wrapper, and aggregator_params
    aggregator_params = aggregator_params or {}
    return aggregator_fn(outputs, agreement_wrapper, question, **aggregator_params)
