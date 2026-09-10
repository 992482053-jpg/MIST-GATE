from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer, normalize
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from .MISTGATE import MISTGATE
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def train_MISTGATE(adata1, adata2, hidden_dims=[512, 30], bp_width=450, temp=1.0, type='ATAC', n_epochs=500, lr=0.0001, key_added='MISTGATE',
                    gradient_clipping=5, nonlinear=True, weight_decay=0.0001, verbose=False,
                    random_seed=2020, pre_labels=None, pre_resolution=0.2,
                    save_attention=False, save_loss=False, save_reconstrction=False, protein_value=0.001,
                    use_wnn_graph=True, refine_spatial=True, wnn_neighbors=15, wnn_pca_dim=30,
                    refined_graph_weight=0.5, refined_graph_mode='soft', dual_graph_weight=0.3,
                    triplet_lambda=0.1, triplet_margin=0.5, triplet_neighbors=3,
                    triplet_farthest_ratio=0.6, max_triplets=20000,
                    consistency_lambda=0.05):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.

    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    nonlinear
        If True, the nonlinear avtivation is performed.
    weight_decay
        Weight decay for AdamOptimizer.
    pre_labels
        The key in adata.obs for the manually designate the pre-clustering results. Only used when alpha>0.
    pre_resolution
        The resolution parameter of sc.tl.louvain for the pre-clustering. Only used when alpha>0 and per_labels==None.
    save_attention
        If True, the weights of the attention layers are saved in adata.uns['STAGATE_attention']
    save_loss
        If True, the training loss is saved in adata.uns['STAGATE_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STAGATE_ReX'].

    Returns
    -------
    AnnData
    """

    tf.reset_default_graph()
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    if 'highly_variable' in adata1.var.columns and 'highly_variable' in adata2.var.columns:
        adata_Vars1 = adata1[:, adata1.var['highly_variable']]
        adata_Vars2 = adata2[:, adata2.var['highly_variable']]
    else:
        adata_Vars1 = adata1
        adata_Vars2 = adata2
    # import pandas as pd
    if isinstance(adata_Vars1.X, np.ndarray):
        X1 = pd.DataFrame(
            adata_Vars1.X[:, ], index=adata_Vars1.obs.index, columns=adata_Vars1.var.index)
    else:
        X1 = pd.DataFrame(adata_Vars1.X.toarray()[
                          :, ], index=adata_Vars1.obs.index, columns=adata_Vars1.var.index)
    if isinstance(adata_Vars2.X, np.ndarray):
        X2 = pd.DataFrame(
            adata_Vars2.X[:, ], index=adata_Vars2.obs.index, columns=adata_Vars2.var.index)
    else:
        X2 = pd.DataFrame(adata_Vars2.X.toarray()[
                          :, ], index=adata_Vars2.obs.index, columns=adata_Vars2.var.index)
#    X1 = pd.DataFrame(adata_Vars1.X.toarray()[:, ], index=adata_Vars1.obs.index, columns=adata_Vars1.var.index)
#    X2 = pd.DataFrame(adata_Vars2.X.toarray()[:, ], index=adata_Vars2.obs.index, columns=adata_Vars2.var.index)
    if verbose:
        print('Size of Input of rna: ', adata_Vars1.shape)
        print('Size of Input of atac: ', adata_Vars2.shape)
    cells = np.array(X1.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    genes = np.array(X1.columns)
    peaks = np.array(X2.columns)
    genes_id_tran = dict(zip(genes, range(genes.shape[0])))
    peaks_id_tran = dict(zip(peaks, range(peaks.shape[0])))
    if 'Spatial_Net' not in adata1.uns.keys():
        raise ValueError(
            "Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    Spatial_Net = adata_Vars1.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(
        adata_Vars1.n_obs, adata_Vars1.n_obs))

    if use_wnn_graph:
        feature_G, feature_embedding = build_wnn_feature_graph(
            X1.values, X2.values, n_neighbors=wnn_neighbors,
            pca_dim=wnn_pca_dim, random_seed=random_seed)
        if refine_spatial:
            G_train = refine_spatial_graph(
                G, feature_G, weight=refined_graph_weight,
                mode=refined_graph_mode)
        else:
            G_train = G
        G_feature_tf = prepare_graph_data(feature_G)
    else:
        feature_embedding = build_joint_feature_embedding(
            X1.values, X2.values, pca_dim=wnn_pca_dim,
            random_seed=random_seed)
        G_train = G
        G_feature_tf = prepare_graph_data(G)

    G_tf = prepare_graph_data(G_train)
    if verbose and use_wnn_graph:
        print('Feature graph contains %d edges.' % feature_G.nnz)
        print('Training spatial graph contains %d edges.' % G_train.nnz)

    if 'gene_peak_Net' not in adata1.uns.keys():
        raise ValueError(
            "gene_peak_Net is not existed! Run Cal_gene_peak_Net first!")
    gene_peak_Net = adata_Vars1.uns['gene_peak_Net']
    # gene_peak_Net.columns = ['Gene','Peak']
    if type == 'protein':
        gene_peak_Net.columns = ['Gene','Peak']
    G_gp_df = gene_peak_Net.copy()
    G_gp_df['Gene'] = G_gp_df['Gene'].map(genes_id_tran)
    G_gp_df['Peak'] = G_gp_df['Peak'].map(peaks_id_tran) + adata_Vars1.n_vars
    if type == 'ATAC':
        dist = G_gp_df['Distance']
        weights = np.concatenate((((dist + bp_width) / bp_width) **
                                 (-0.75), ((dist + bp_width) / bp_width) ** (-0.75)), axis=0)
        temp=-10
        G_gp = sp.coo_matrix((weights, (np.concatenate((G_gp_df['Gene'], G_gp_df['Peak']), axis=0), np.concatenate((G_gp_df['Peak'], G_gp_df['Gene']), axis=0))), shape=(adata_Vars1.n_vars+adata_Vars2.n_vars,
                                                                                                                                                                          adata_Vars1.n_vars+adata_Vars2.n_vars))
    elif type == 'ATAC_RNA':
        dist = G_gp_df['Distance']
        bp_width=2000
        weights = np.concatenate((((dist + bp_width) / bp_width) **
                                    (-0.75), ((dist + bp_width) / bp_width) ** (-0.75)), axis=0)
        G_gp = sp.coo_matrix((weights, (np.concatenate((G_gp_df['Gene'], G_gp_df['Peak']), axis=0), np.concatenate((G_gp_df['Peak'], G_gp_df['Gene']), axis=0))), shape=(adata_Vars1.n_vars+adata_Vars2.n_vars,
                                                                                                                                                                            adata_Vars1.n_vars+adata_Vars2.n_vars))
    else:
        G_gp = sp.coo_matrix((np.ones(G_gp_df.shape[0]*2)*protein_value, (np.concatenate((G_gp_df['Gene'], G_gp_df['Peak']), axis=0), np.concatenate((G_gp_df['Peak'], G_gp_df['Gene']), axis=0))), shape=(adata_Vars1.n_vars+adata_Vars2.n_vars,
                                                                                                                                                                                             adata_Vars1.n_vars+adata_Vars2.n_vars))
   
    G_gp_tf = prepare_graph_data(G_gp)

    if triplet_lambda > 0:
        triplet_samples = construct_mnn_triplets(
            feature_embedding, n_neighbors=triplet_neighbors,
            farthest_ratio=triplet_farthest_ratio,
            max_triplets=max_triplets, random_seed=random_seed)
        if verbose:
            print('Constructed %d MNN triplets for metric learning.' % triplet_samples[0].shape[0])
    else:
        triplet_samples = (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    trainer = MISTGATE(hidden_dims1=[X1.shape[1]] + hidden_dims, hidden_dims2=[X2.shape[1]] + hidden_dims, spot_num=X1.shape[0],
                        temp=temp, n_epochs=n_epochs, lr=lr, gradient_clipping=gradient_clipping,
                        nonlinear=nonlinear, weight_decay=weight_decay, verbose=verbose,
                        random_seed=random_seed,
                        dual_graph_weight=dual_graph_weight if use_wnn_graph else 0.0,
                        triplet_lambda=triplet_lambda,
                        triplet_margin=triplet_margin,
                        consistency_lambda=consistency_lambda)

    trainer(G_tf, G_feature_tf, G_gp_tf, X1, X2, triplet_samples=triplet_samples)

    loss_df = pd.DataFrame({
        'loss': trainer.loss_list,
        'loss_atac': trainer.loss_list_atac,
        'loss_rna': trainer.loss_list_rna,
        'loss_clip': trainer.loss_list_clip,
        'loss_triplet': trainer.loss_list_triplet,
        'loss_consistency': trainer.loss_list_consistency,
        'weight_decay_loss': trainer.weight_decay_loss_list,
    })

    # loss_df.to_csv('losses.csv', index=False)
    embeddings_RNA, embeddings_ATAC, attentions1, attentions2, attentions_gp, loss, ReX_RNA, ReX_ATAC = trainer.infer(G_tf, G_feature_tf,
                                                                                                                      G_gp_tf, X1, X2)

    # embeddings, attentions, loss, ReX
    cell_reps1 = pd.DataFrame(embeddings_RNA)
    cell_reps1.index = cells

    adata1.obsm[key_added] = cell_reps1.loc[adata1.obs_names, ].values

    cell_reps2 = pd.DataFrame(embeddings_ATAC)
    cell_reps2.index = cells

    adata2.obsm[key_added] = cell_reps2.loc[adata2.obs_names, ].values
    norm2 = Normalizer(norm='l2')

    adata1.obsm[key_added + '_clip_all'] = (norm2.fit_transform(cell_reps1.loc[adata1.obs_names,].values) +
                                            norm2.fit_transform(cell_reps2.loc[adata2.obs_names, ].values)) / 2.0
    adata2.obsm[key_added + '_clip_all'] = (norm2.fit_transform(cell_reps1.loc[adata1.obs_names,].values) +
                                            norm2.fit_transform(cell_reps2.loc[adata2.obs_names, ].values)) / 2.0
    if save_attention:
        adata1.uns['MISTGATE_attention'] = attentions1
        adata2.uns['MISTGATE_attention'] = attentions2
        adata1.uns['MISTGATE_gene_peak_attention'] = attentions_gp
        adata2.uns['MISTGATE_gene_peak_attention'] = attentions_gp
    if save_loss:
        adata1.uns['MISTGATE_loss_united'] = loss
        adata2.uns['MISTGATE_loss_united'] = loss
        adata1.uns['MISTGATE_loss_detail'] = loss_df
        adata2.uns['MISTGATE_loss_detail'] = loss_df
    if save_reconstrction:
        ReX_RNA = pd.DataFrame(ReX_RNA, index=X1.index, columns=X1.columns)
        ReX_RNA[ReX_RNA < 0] = 0
        adata1.layers['MISTGATE_ReX'] = ReX_RNA.values

        ReX_ATAC = pd.DataFrame(ReX_ATAC, index=X2.index, columns=X2.columns)
        ReX_ATAC[ReX_ATAC < 0] = 0
        adata2.layers['MISTGATE_ReX'] = ReX_ATAC.values
    return adata1, adata2


def prune_spatial_Net(Graph_df, label):
    print('------Pruning the graph...')
    print('%d edges before pruning.' % Graph_df.shape[0])
    pro_labels_dict = dict(zip(list(label.index), label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label']
                            == Graph_df['Cell2_label'],]
    print('%d edges after pruning.' % Graph_df.shape[0])
    return Graph_df


def build_joint_feature_embedding(X1, X2, pca_dim=30, random_seed=2020):
    X1 = _pca_normalize(X1, pca_dim, random_seed)
    X2 = _pca_normalize(X2, pca_dim, random_seed)
    return normalize(np.concatenate([X1, X2], axis=1), norm='l2')


def build_wnn_feature_graph(X1, X2, n_neighbors=15, pca_dim=30, random_seed=2020):
    """Build a lightweight WNN-style feature graph for spot-level dual-graph learning."""
    Z1 = _pca_normalize(X1, pca_dim, random_seed)
    Z2 = _pca_normalize(X2, pca_dim, random_seed)
    A1, dist1 = _knn_affinity_graph(Z1, n_neighbors)
    A2, dist2 = _knn_affinity_graph(Z2, n_neighbors)

    local1 = np.maximum(np.mean(dist1[:, 1:], axis=1), 1e-8)
    local2 = np.maximum(np.mean(dist2[:, 1:], axis=1), 1e-8)
    score1 = 1.0 / local1
    score2 = 1.0 / local2
    w1 = score1 / (score1 + score2 + 1e-8)
    w2 = 1.0 - w1

    W1 = sp.diags(w1.astype(np.float32))
    W2 = sp.diags(w2.astype(np.float32))
    A = W1.dot(A1) + W2.dot(A2)
    A = _symmetrize_max(A)
    A.eliminate_zeros()
    embedding = normalize(np.concatenate([w1[:, None] * Z1, w2[:, None] * Z2], axis=1), norm='l2')
    return A.tocoo(), embedding


def refine_spatial_graph(spatial_graph, feature_graph, weight=0.5, mode='soft'):
    """Refine spatial edges with feature/WNN similarity, following SpaMICS' graph refinement idea."""
    spatial_graph = spatial_graph.tocsr().astype(np.float32)
    feature_graph = feature_graph.tocsr().astype(np.float32)
    weight = float(np.clip(weight, 0.0, 1.0))

    if mode == 'intersection':
        feature_binary = feature_graph.copy()
        feature_binary.data = np.ones_like(feature_binary.data)
        refined = spatial_graph.multiply(feature_binary)
        if refined.nnz < spatial_graph.shape[0]:
            refined = (1.0 - weight) * spatial_graph + weight * spatial_graph.multiply(feature_graph)
    else:
        feature_on_spatial = spatial_graph.multiply(feature_graph)
        refined = (1.0 - weight) * spatial_graph + weight * feature_on_spatial

    refined = _symmetrize_max(refined)
    refined.eliminate_zeros()
    return refined.tocoo()


def construct_mnn_triplets(embedding, n_neighbors=3, farthest_ratio=0.6,
                           max_triplets=20000, random_seed=2020):
    """Construct SMART-style MNN triplets from a joint feature embedding."""
    rng = np.random.RandomState(random_seed)
    embedding = normalize(np.asarray(embedding, dtype=np.float32), norm='l2')
    n_obs = embedding.shape[0]
    if n_obs < 3:
        empty = np.array([], dtype=np.int32)
        return empty, empty, empty

    n_neighbors = max(1, min(int(n_neighbors), n_obs - 1))
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
    nn.fit(embedding)
    _, indices = nn.kneighbors(embedding)
    neighbor_indices = indices[:, 1:]
    neighbor_sets = [set(row) for row in neighbor_indices]

    pairs = []
    for anchor, neighbors in enumerate(neighbor_indices):
        for positive in neighbors:
            if anchor in neighbor_sets[positive]:
                pairs.append((anchor, positive))

    if not pairs:
        pairs = [(i, neighbor_indices[i, 0]) for i in range(n_obs)]

    if max_triplets is not None and max_triplets > 0 and len(pairs) > max_triplets:
        keep = rng.choice(len(pairs), size=max_triplets, replace=False)
        pairs = [pairs[i] for i in keep]

    anchors, positives, negatives = [], [], []
    farthest_ratio = float(np.clip(farthest_ratio, 0.05, 1.0))
    for anchor, positive in pairs:
        distances = np.linalg.norm(embedding - embedding[anchor], axis=1)
        distances[anchor] = -np.inf
        distances[positive] = -np.inf
        valid_count = max(1, n_obs - 2)
        n_far = max(1, int(valid_count * farthest_ratio))
        candidate_idx = np.argpartition(distances, -n_far)[-n_far:]
        candidate_idx = candidate_idx[np.isfinite(distances[candidate_idx])]
        if candidate_idx.size == 0:
            continue
        negative = rng.choice(candidate_idx)
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)

    return (np.asarray(anchors, dtype=np.int32),
            np.asarray(positives, dtype=np.int32),
            np.asarray(negatives, dtype=np.int32))


def _pca_normalize(X, pca_dim=30, random_seed=2020):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    max_components = min(X.shape[0] - 1, X.shape[1])
    if pca_dim is not None and max_components > 1 and X.shape[1] > pca_dim:
        n_components = min(int(pca_dim), max_components)
        X = PCA(n_components=n_components, random_state=random_seed).fit_transform(X)
    return normalize(X, norm='l2')


def _knn_affinity_graph(embedding, n_neighbors=15):
    n_obs = embedding.shape[0]
    k = max(1, min(int(n_neighbors), n_obs - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(embedding)
    distances, indices = nn.kneighbors(embedding)
    local_sigma = np.maximum(np.median(distances[:, 1:], axis=1), 1e-8)

    rows = np.repeat(np.arange(n_obs), k)
    cols = indices[:, 1:].reshape(-1)
    dist = distances[:, 1:].reshape(-1)
    sigma = np.repeat(local_sigma, k)
    values = np.exp(-dist / (sigma + 1e-8)).astype(np.float32)
    graph = sp.coo_matrix((values, (rows, cols)), shape=(n_obs, n_obs))
    graph = _symmetrize_max(graph)
    graph.eliminate_zeros()
    return graph.tocoo(), distances


def _symmetrize_max(graph):
    graph = graph.tocsr()
    return graph.maximum(graph.T).tocoo()


def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    # data =  adj.tocoo().data
    # adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape)


def recovery_Imputed_Count(adata, size_factor):
    assert ('ReX' in adata.uns)
    temp_df = adata.uns['ReX'].copy()
    sf = size_factor.loc[temp_df.index]
    temp_df = np.expm1(temp_df)
    temp_df = (temp_df.T * sf).T
    adata.uns['ReX_Count'] = temp_df
    return adata


# Backward-compatible function name for old notebooks.
train_MultiGATE = train_MISTGATE
