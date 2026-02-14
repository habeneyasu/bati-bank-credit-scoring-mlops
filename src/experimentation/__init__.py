"""
Experimentation Module

Provides A/B testing framework for model comparison.
"""

from src.experimentation.ab_testing import (
    ABTestingFramework,
    TrafficSplitter,
    StatisticalAnalyzer,
    get_ab_testing_framework
)

__all__ = [
    "ABTestingFramework",
    "TrafficSplitter",
    "StatisticalAnalyzer",
    "get_ab_testing_framework",
]
