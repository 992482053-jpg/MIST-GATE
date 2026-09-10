"""Public API for MIST-GATE.

Use ``MISTGATE`` as the Python package name and ``MIST-GATE`` in figures and
manuscript text.
"""

__version__ = "0.2.0"

from .MISTGATE import MISTGATE
from .Train_MISTGATE import (
    build_wnn_feature_graph,
    construct_mnn_triplets,
    refine_spatial_graph,
    train_MISTGATE,
)
from .model_MISTGATE import MISTGATEModel
from .utils import (
    Cal_Spatial_Net,
    Cal_gene_peak_Net,
    Cal_gene_peak_Net_new,
    Cal_gene_protein_Net,
    Stats_Spatial_Net,
    mclust_python,
    wnn_python,
)

# Transitional aliases for old notebooks. New code should use the names above.
train_MultiGATE = train_MISTGATE
MultiGATE = MISTGATE
MGATE = MISTGATEModel

__all__ = [
    "MISTGATE",
    "MISTGATEModel",
    "train_MISTGATE",
    "build_wnn_feature_graph",
    "refine_spatial_graph",
    "construct_mnn_triplets",
    "Cal_Spatial_Net",
    "Cal_gene_peak_Net",
    "Cal_gene_peak_Net_new",
    "Cal_gene_protein_Net",
    "Stats_Spatial_Net",
    "wnn_python",
    "mclust_python",
]
