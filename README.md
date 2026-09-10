## MIST-GATE
MIST-GATE (Multi-omics Integrated Spatial Triplet Graph Attention Encoder)** is a spatial multi-omics integration framework designed for **robust spatial-domain identification** and **cross-modality regulatory inference.

MIST-GATE jointly models spatial structural information and multi-modal molecular features to achieve effective integration across different omics modalities. The current framework consists of the following core components:

- a **Python-based multi-modal feature graph**;
- **feature-guided spatial graph refinement**;
- **dual spatial/feature graph learning**;
- **MNN-based triplet metric learning**;
- **Python-based graph clustering and reproducible ARI/ICC evaluation**.

In **manuscripts, framework figures, and other presentation materials**, the model name should be written as **MIST-GATE**.

In **Python code**, the identifier **MISTGATE** should be used consistently:

```python
import MISTGATE

adata_rna, adata_second = MISTGATE.train_MISTGATE(
    adata_rna,
    adata_second,
    type="ATAC_RNA",
)
```

New **AnnData fields, model results, and output files** use the `MISTGATE` prefix.

## Project Structure

- `MISTGATE/`: implementation of the core MIST-GATE model and related functionalities.
- `Reproduce/clustering/`: spatial clustering and spatial-domain identification experiments of MIST-GATE across different spatial multi-omics datasets.
- `Reproduce/cis-regulation/`: peak–gene, eQTL, and cross-modality regulatory inference analyses and evaluations using MIST-GATE.

## MIST-GATE Workflow

MIST-GATE first constructs a **multi-modal feature graph** based on molecular features from different omics modalities and uses feature information to refine the original spatial neighborhood relationships. It then employs a **dual-graph learning strategy**, jointly learning from the spatial graph and feature graph to capture both spatial structural information and molecular feature representations.

On this basis, **MNN (Mutual Nearest Neighbors)-based triplet metric learning** is introduced to enhance the alignment of spatial locations with consistent biological characteristics across modalities in the latent representation space, while increasing the separation between dissimilar samples. This process enables MIST-GATE to learn a more stable and discriminative joint representation.

The resulting joint representation can be used for downstream tasks including **spatial-domain identification, spatial clustering, cross-modality consistency analysis, and regulatory inference**. Model performance can be quantitatively evaluated using metrics such as **Adjusted Rand Index (ARI)** and **Intraclass Correlation Coefficient (ICC)**.