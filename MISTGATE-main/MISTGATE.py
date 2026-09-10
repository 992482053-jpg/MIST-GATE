import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.sparse as sp
import numpy as np
import os
from .model_MISTGATE import MISTGATEModel
from tqdm import tqdm

class MISTGATE():

    def __init__(self, hidden_dims1, hidden_dims2, spot_num, temp, n_epochs=500, lr=0.0001,
                 gradient_clipping=5, nonlinear=True, weight_decay=0.0001,
                 verbose=False, random_seed=2020, config=None,
                 dual_graph_weight=0.0, triplet_lambda=0.0,
                 triplet_margin=0.5, consistency_lambda=0.0):
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        self.loss_list_rna = []
        self.loss_list_atac = []
        self.loss_list = []
        self.loss_list_clip = []
        self.loss_list_triplet = []
        self.loss_list_consistency = []
        self.weight_decay_loss_list = []
        self.lr = lr
        self.n_epochs = n_epochs
        self.gradient_clipping = gradient_clipping
        self.build_placeholders()
        self.verbose = verbose
        self.config = config
        
        self.temp = temp
        self.mgate = MISTGATEModel(hidden_dims1, hidden_dims2, spot_num, temp, nonlinear, weight_decay,
                           dual_graph_weight=dual_graph_weight,
                           triplet_lambda=triplet_lambda,
                           triplet_margin=triplet_margin,
                           consistency_lambda=consistency_lambda)
        self.loss, self.loss_rna, self.loss_atac, self.weight_decay_loss, self.clip_loss, \
        self.triplet_loss, self.consistency_loss, self.H1, self.H2, self.C1, \
        self.C2, self.Cgp, self.ReX1, self.ReX2 = self.mgate(
            self.A, self.prune_A, self.GP, self.X1, self.X2,
            self.triplet_anchors, self.triplet_positives, self.triplet_negatives)
        self.optimize(self.loss)
        self.build_session()
        # self.hidden_dims = hidden_dims

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.prune_A = tf.sparse_placeholder(dtype=tf.float32)
        self.GP = tf.sparse_placeholder(dtype=tf.float32)
        self.X1 = tf.placeholder(dtype=tf.float32)
        self.X2 = tf.placeholder(dtype=tf.float32)
        self.triplet_anchors = tf.placeholder(dtype=tf.int32, shape=[None])
        self.triplet_positives = tf.placeholder(dtype=tf.int32, shape=[None])
        self.triplet_negatives = tf.placeholder(dtype=tf.int32, shape=[None])

    def build_session(self, gpu=True):
        if self.config is not None:
            # Use the provided configuration
            config = self.config
        else:
            # Use default configuration
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # Add A100 compatibility options
            config.gpu_options.per_process_gpu_memory_fraction = 0.7
            config.graph_options.rewrite_options.memory_optimization = True
            config.graph_options.rewrite_options.arithmetic_optimization = True

        deterministic = os.environ.get(
            "MISTGATE_DETERMINISTIC",
            os.environ.get("MULTIGATE_DETERMINISTIC", "0"),
        )
        if deterministic == "1":
            config.intra_op_parallelism_threads = 1
            config.inter_op_parallelism_threads = 1
            
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def __call__(self, A, prune_A, GP, X1, X2, triplet_samples=None):
        # for epoch in tqdm(range(self.n_epochs)):
        #     self.run_epoch(epoch, A, prune_A, GP, X1, X2)
        
        with tqdm(total=self.n_epochs, desc="Epoch Progress", unit="epoch") as pbar:
            for epoch in range(self.n_epochs):
                loss = self.run_epoch(epoch, A, prune_A, GP, X1, X2, triplet_samples)
                pbar.update(1)
                if self.verbose:
                    tqdm.write(f"Epoch: {epoch}, Loss: {loss:.4f}")
                    
    def run_epoch(self, epoch, A, prune_A, GP, X1, X2, triplet_samples=None):
        if triplet_samples is None:
            triplet_samples = (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )
        anchors, positives, negatives = triplet_samples
        try:
            loss, loss_rna, loss_atac, weight_decay_loss, clip_loss, triplet_loss, consistency_loss, _ = self.session.run(
                                             [self.loss, self.loss_rna, self.loss_atac, self.weight_decay_loss,
                                              self.clip_loss, self.triplet_loss, self.consistency_loss, self.train_op],
                                             feed_dict={self.A: A,
                                                        self.prune_A: prune_A,
                                                        self.GP: GP,
                                                        self.X1: X1,
                                                        self.X2: X2,
                                                        self.triplet_anchors: anchors,
                                                        self.triplet_positives: positives,
                                                        self.triplet_negatives: negatives})
            self.loss_list.append(loss)
            self.loss_list_atac.append(loss_atac)
            self.loss_list_rna.append(loss_rna)
            self.loss_list_clip.append(clip_loss)
            self.loss_list_triplet.append(triplet_loss)
            self.loss_list_consistency.append(consistency_loss)
            self.weight_decay_loss_list.append(weight_decay_loss)
            return loss
        except tf.errors.InternalError as e:
            if "Blas GEMM launch failed" in str(e):
                print("BLAS GEMM operation failed. Attempting with smaller batch or reduced precision...")
                # Try to continue with a gentler operation approach
                with tf.device('/cpu:0'):  # Fall back to CPU if GEMM fails on GPU
                    loss, loss_rna, loss_atac, weight_decay_loss, clip_loss, triplet_loss, consistency_loss, _ = self.session.run(
                                                 [self.loss, self.loss_rna, self.loss_atac, self.weight_decay_loss,
                                                  self.clip_loss, self.triplet_loss, self.consistency_loss, self.train_op],
                                                 feed_dict={self.A: A,
                                                            self.prune_A: prune_A,
                                                            self.GP: GP,
                                                            self.X1: X1,
                                                            self.X2: X2,
                                                            self.triplet_anchors: anchors,
                                                            self.triplet_positives: positives,
                                                            self.triplet_negatives: negatives})
                    self.loss_list.append(loss)
                    self.loss_list_atac.append(loss_atac)
                    self.loss_list_rna.append(loss_rna)
                    self.loss_list_clip.append(clip_loss)
                    self.loss_list_triplet.append(triplet_loss)
                    self.loss_list_consistency.append(consistency_loss)
                    self.weight_decay_loss_list.append(weight_decay_loss)
                    return loss
            else:
                # If it's not a BLAS error, re-raise
                raise

    def infer(self, A, prune_A, GP, X1, X2):
        H1, H2, C1, C2, Cgp, ReX1, ReX2 = self.session.run([self.H1, self.H2, self.C1, self.C2, self.Cgp, self.ReX1, self.ReX2],
                           feed_dict={self.A: A,
                                      self.prune_A: prune_A,
                                      self.GP: GP,
                                      self.X1: X1,
                                      self.X2: X2,
                                      self.triplet_anchors: np.array([], dtype=np.int32),
                                      self.triplet_positives: np.array([], dtype=np.int32),
                                      self.triplet_negatives: np.array([], dtype=np.int32)})

        return H1, H2, self.Conbine_Atten_l(C1), self.Conbine_Atten_l(C2), self.Conbine_Atten_l(Cgp), self.loss_list, ReX1, ReX2

    def Conbine_Atten_l(self, input):
        return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])), shape=(input[layer][2][0], input[layer][2][1])) for layer in input]


# Backward-compatible class name for serialized notebooks.
MultiGATE = MISTGATE
   
