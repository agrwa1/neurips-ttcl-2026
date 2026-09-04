"""Deterministic generation of the paper figures."""

from .bias_accumulation import generate_bias_accumulation_figure
from .canonical_position import generate_canonical_position_figure
from .flow_routing import generate_flow_routing_figures
from .method_comparison import generate_method_comparison_figures

__all__ = [
    "generate_bias_accumulation_figure",
    "generate_canonical_position_figure",
    "generate_flow_routing_figures",
    "generate_method_comparison_figures",
]
