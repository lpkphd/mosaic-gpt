"""
MOSAIC-GPT: Modular Open-Source Architecture for Intelligent Computation

A fully modular, sub-500M parameter language model combining state-of-the-art
techniques (MLA, MoE, SwiGLU, RoPE, RMSNorm) where every component is swappable.
"""

from mosaic.config import MosaicConfig
from mosaic.model import MosaicGPT

__version__ = "0.1.0"
__all__ = ["MosaicConfig", "MosaicGPT"]
