#!/usr/bin/env python
# coding: utf-8

# This Jupyter Notebook includes code adapted from the RouteNet project.
# 
# Original author:
# 
# Krzysztof Rusek AGH University of Science and Technology, Department of Communications, Krakow, Poland. Email: krusek@agh.edu.pl This code is licensed under the BSD 3-Clause License. See the LICENSE file in this repository for details.

import tensorflow as tf

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from tensorflow import keras

import tensorflow_addons as tfa

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import glob


def parse(serialized, target='delay'): #Target is the name of predicted variable
    
    with tf.name_scope('parse'):    
        features = tf.compat.v1.parse_single_example(
            serialized,
            features={
                'traffic':tf.compat.v1.VarLenFeature(tf.float32),
                target:tf.compat.v1.VarLenFeature(tf.float32),
                'links':tf.compat.v1.VarLenFeature(tf.int64),
                'paths':tf.compat.v1.VarLenFeature(tf.int64),
                'sequances':tf.compat.v1.VarLenFeature(tf.int64),
                'n_links':tf.compat.v1.FixedLenFeature([],tf.int64), 
                'n_paths':tf.compat.v1.FixedLenFeature([],tf.int64),
                'n_total':tf.compat.v1.FixedLenFeature([],tf.int64)
            }
        )
        
        for k in ['traffic',target,'links','paths','sequances']:
            features[k] = tf.compat.v1.sparse_tensor_to_dense(features[k])
            if k == 'delay':
                # features[k] = (features[k]-2.8)/2.5 #nsfnet
                features[k] = (features[k]-7.4)/7.0 #gbn
                
            if k == 'traffic':
                #features[k] = (features[k]-0.76)/.008
                features[k] = (features[k]-0.5)/.5
                
            # if k == 'drops':
            #     features[k] = (features[k])/12000/(0.5*features['traffic']+0.5) #loss rate
            #if k == 'jitter':
                #features[k] = (tf.math.log( features[k] )-2.0)/2.0 #logjitter
            
    return {k:v for k,v in features.items() if k is not target },features[target]


def tfrecord_input_fn(filenames,hparams,shuffle_buf=1000, target='delay'):
    
    files = tf.data.Dataset.from_tensor_slices(filenames)
    files = files.shuffle(len(filenames))

    ds = files.interleave(tf.data.TFRecordDataset, cycle_length=4)

    if shuffle_buf:
        ds = ds.shuffle(shuffle_buf).repeat()
    
    # ds = ds.map(lambda buf:parse(buf,target), num_parallel_calls=2)
    ds = ds.map(lambda buf:parse(buf,target), num_parallel_calls=tf.data.AUTOTUNE)

    shapes=(
        {
        'traffic':[hparams.node_count*(hparams.node_count-1)],
        'links':[-1],
        'paths':[-1],
        'sequances':[-1],
        'n_links':[],
        'n_paths':[],
        'n_total':[]
        },
        [hparams.node_count*(hparams.node_count-1)]
    )
    
    ds = ds.padded_batch(hparams.batch_size,shapes)
    ds = ds.prefetch(1)
    
    return ds


class ComnetModel(tf.keras.Model):
    def __init__(self,hparams, output_units=1):
        super(ComnetModel, self).__init__()
        self.hparams = hparams

        self.edge_update = tf.compat.v1.nn.rnn_cell.GRUCell(hparams.link_state_dim, dtype=tf.float32)
        self.path_update = tf.compat.v1.nn.rnn_cell.GRUCell(hparams.path_state_dim, dtype=tf.float32)
        # self.edge_update = tf.keras.layers.GRUCell(hparams.link_state_dim)
        # self.path_update = tf.keras.layers.GRUCell(hparams.path_state_dim)
        
        self.readout = tf.keras.models.Sequential()
        
        self.readout.add(keras.layers.Dense(hparams.readout_units, activation=tf.nn.selu, kernel_regularizer=tf.keras.regularizers.l2(hparams.l2)))
        self.readout.add(keras.layers.Dropout(rate=hparams.dropout_rate))
        
        self.readout.add(keras.layers.Dense(hparams.readout_units, activation=tf.nn.selu, kernel_regularizer=tf.keras.regularizers.l2(hparams.l2)))
        self.readout.add(keras.layers.Dropout(rate=hparams.dropout_rate))
        
        self.readout.add(keras.layers.Dense(output_units, kernel_regularizer=tf.keras.regularizers.l2(hparams.l2)))
            
    def build(self, input_shape=None):
        del input_shape
        self.edge_update.build(tf.TensorShape([None,self.hparams.path_state_dim]))
        self.path_update.build(tf.TensorShape([None,self.hparams.link_state_dim]))
        self.readout.build(input_shape = [None,self.hparams.path_state_dim])
        self.built = True

    def call(self, inputs, training=False):
        f_ = inputs
        shape = tf.stack([f_['n_links'],self.hparams.link_state_dim], axis=0)
        link_state = tf.zeros(shape)
        shape = tf.stack([f_['n_paths'],self.hparams.path_state_dim-1], axis=0)
        path_state = tf.concat([tf.expand_dims(f_['traffic'],axis=1), tf.zeros(shape)], axis=1)

        links = f_['links'][0:f_["n_total"]]
        paths = f_['paths'][0:f_["n_total"]]
        seqs=  f_['sequances'][0:f_["n_total"]]
        
        for _ in range(self.hparams.T):
        
            h_tild = tf.gather(link_state,links)

            ids=tf.stack([paths, seqs], axis=1)            
            max_len = tf.reduce_max(seqs)+1
            shape = tf.stack([f_['n_paths'], max_len, self.hparams.link_state_dim])
            lens = tf.compat.v1.segment_sum(data=tf.ones_like(paths), segment_ids=paths)

            link_inputs = tf.scatter_nd(ids, h_tild, shape)
            outputs, path_state = tf.compat.v1.nn.dynamic_rnn(self.path_update,link_inputs,sequence_length=lens,initial_state=path_state,dtype=tf.float32)
            m = tf.gather_nd(outputs,ids)
            m = tf.compat.v1.unsorted_segment_sum(m, links ,f_['n_links'])
            _,link_state = self.edge_update(m, link_state)
        
        r = self.readout(path_state,training=training)
        
        return r


def streaming_pearson_correlation_dl(labels, predictions, weights=None):
    """
    Compute streaming Pearson correlation with distributed training compatibility.
    Handles both single-replica and multi-replica contexts.
    Returns (value_tensor, update_op) for use in eval_metric_ops.
    """
    global learning_strategy
    
    # Cast inputs to float32 for compatibility
    labels = tf.cast(labels, tf.float32)
    predictions = tf.cast(predictions, tf.float32)

    # Batch-wise computations
    batch_size = tf.cast(tf.size(labels), tf.float32)
    batch_mean_x = tf.reduce_mean(predictions)
    batch_mean_y = tf.reduce_mean(labels)
    batch_mean_x_squared = tf.reduce_mean(tf.square(predictions))
    batch_mean_y_squared = tf.reduce_mean(tf.square(labels))
    batch_mean_xy = tf.reduce_mean(predictions * labels)

    # Use tf.metrics to accumulate streaming values
    mean_x, update_mean_x = tf.compat.v1.metrics.mean(batch_mean_x, weights)
    mean_y, update_mean_y = tf.compat.v1.metrics.mean(batch_mean_y, weights)
    mean_x_squared, update_mean_x_squared = tf.compat.v1.metrics.mean(batch_mean_x_squared, weights)
    mean_y_squared, update_mean_y_squared = tf.compat.v1.metrics.mean(batch_mean_y_squared, weights)
    mean_xy, update_mean_xy = tf.compat.v1.metrics.mean(batch_mean_xy, weights)
    count, update_count = tf.compat.v1.metrics.mean(batch_size, weights)

    # Compute Pearson correlation
    def compute_correlation():
        covariance = mean_xy - mean_x * mean_y
        variance_x = mean_x_squared - tf.square(mean_x)
        variance_y = mean_y_squared - tf.square(mean_y)
        denominator = tf.sqrt(variance_x * variance_y)
        return tf.where(tf.greater(denominator, 0), covariance / denominator, 0.0)

    rho = compute_correlation()

    # Group all update ops
    update_op = tf.group(
        update_mean_x, update_mean_y, update_mean_x_squared,
        update_mean_y_squared, update_mean_xy, update_count
    )

    # Distributed training compatibility
    if learning_strategy and learning_strategy.num_replicas_in_sync > 1:
        
        # Aggregate distributed values using the strategy context
        def aggregate_rho(learning_strategy, rho):
            return learning_strategy.reduce(tf.distribute.ReduceOp.MEAN, rho, axis=None)

        # Use merge_call to enter cross-replica context
        rho = tf.distribute.get_replica_context().merge_call(lambda ctx: aggregate_rho(ctx, rho))

    return rho, update_op


def model_fn(features, labels, mode, params):
    
    model = ComnetModel(params)
    model.build()
    #model.summary()

    predictions = tf.map_fn(lambda x: model(x,training=mode==tf.estimator.ModeKeys.TRAIN), features,dtype=tf.float32)
    #predictions = model(features,training=mode==tf.estimator.ModeKeys.TRAIN)
    predictions = tf.squeeze(predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'predictions':predictions})

    loss =  tf.compat.v1.losses.mean_squared_error(labels=labels, predictions = predictions, reduction=tf.compat.v1.losses.Reduction.MEAN)

    regularization_loss = sum(model.losses)
    total_loss = loss + regularization_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops=
            {
                'mse': tf.compat.v1.metrics.mean_squared_error(labels=labels, predictions=predictions),
                #'rho_contrib':tf.contrib.metrics.streaming_pearson_correlation(labels=labels,predictions=predictions),
                'rho': streaming_pearson_correlation_dl(labels, predictions)
            }
        )
    
    assert mode == tf.estimator.ModeKeys.TRAIN

    # Get all trainable variables
    # trainables = model.variables
    trainables = tf.compat.v1.trainable_variables()
    
    grads = tf.gradients(total_loss, trainables)
    grad_var_pairs = zip(grads, trainables)

    # optimizer=tf.compat.v1.train.AdamOptimizer(params.learning_rate)
    optimizer = tfa.optimizers.LAMB(learning_rate=params.learning_rate, weight_decay=params.l2)
    
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    
    # with tf.control_dependencies(update_ops):
    #     train_op = optimizer.apply_gradients(grad_var_pairs, global_step=tf.compat.v1.train.get_global_step())

    global_step = tf.compat.v1.train.get_or_create_global_step()
    apply_op = optimizer.apply_gradients(grad_var_pairs)
    with tf.control_dependencies([apply_op]):
        train_op = tf.compat.v1.assign_add(global_step, 1)
        
    # Log weight and gradient norms
    for var in trainables:
        # Weight norm (parameter size)
        w_norm = tf.norm(var)
        tf.compat.v1.summary.scalar(f"weight_norm/{var.op.name}", w_norm)
     
        # Gradient norm (update strength)
        grad = tf.gradients(loss, var)[0]
        if grad is not None:
            g_norm = tf.norm(grad)
            tf.compat.v1.summary.scalar(f"grad_norm/{var.op.name}", g_norm)
        
    # epsilon for numerical stability
    eps = 1e-7

    # Build list of grad-to-weight ratio tensors
    ratios = []
    for var, g in zip(trainables, grads):
        if g is not None:
            grad_norm = tf.norm(tf.reshape(g, [-1]))
            weight_norm = tf.norm(tf.reshape(var, [-1]))
            ratio = grad_norm / (weight_norm + eps)
            ratios.append(ratio)
            # optional: keep per-layer scalar too
            tf.compat.v1.summary.scalar(f"grad_to_weight/{var.op.name}", ratio)

    # Only aggregate if we have any ratios
    if len(ratios) > 0:
        stacked = tf.stack(ratios)                      # shape: [num_layers]
        mean_ratio = tf.reduce_mean(stacked)
        std_ratio = tf.math.reduce_std(stacked)
        max_ratio = tf.reduce_max(stacked)
        min_ratio = tf.reduce_min(stacked)
        cv_ratio = tf.where(tf.greater(mean_ratio, 0.0), std_ratio / mean_ratio, 0.0)
 
    # Scalars
    tf.compat.v1.summary.scalar("training_loss", loss)
    tf.compat.v1.summary.scalar("regularization_loss", regularization_loss)
    tf.compat.v1.summary.scalar("agg/mean_grad_to_weight", mean_ratio)
    tf.compat.v1.summary.scalar("agg/std_grad_to_weight", std_ratio)
    tf.compat.v1.summary.scalar("agg/cv_grad_to_weight", cv_ratio)
    tf.compat.v1.summary.scalar("agg/max_grad_to_weight", max_ratio)
    tf.compat.v1.summary.scalar("agg/min_grad_to_weight", min_ratio)
    
    # Histograms (weights and gradients)
    hist_summaries = []
    for var in trainables:
        if isinstance(var, tf.Tensor):
            hist_summaries.append(tf.compat.v1.summary.histogram(var.op.name, var))
    for g in grads:
        if g is not None and isinstance(g, tf.Tensor):
            hist_summaries.append(tf.compat.v1.summary.histogram(g.op.name, g))
     
    # Merge summaries
    merged_summary = tf.compat.v1.summary.merge_all()
     
    # Hook
    summary_hook = tf.estimator.SummarySaverHook(save_steps=100, output_dir=log_dir, summary_op=merged_summary)

    hooks = []
    if mode == tf.estimator.ModeKeys.TRAIN:
        hooks = [summary_hook]
    
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=hooks)


def train(args):
    
    print(args.hparams)
    tf.compat.v1.logging.set_verbosity('INFO')

    global learning_strategy, log_dir
    learning_strategy = args.strategy
    log_dir = args.model_dir

    my_checkpointing_config = tf.estimator.RunConfig(
        train_distribute=args.strategy,
        eval_distribute=args.strategy,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=50
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, 
        model_dir=args.model_dir, 
        params=args.hparams, 
        warm_start_from=args.warm, 
        config=my_checkpointing_config)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda:tfrecord_input_fn(args.train,args.hparams,shuffle_buf=args.shuffle_buf,target=args.target),max_steps=args.train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:tfrecord_input_fn(args.eval_,args.hparams,shuffle_buf=None,target=args.target),steps=args.eval_steps,throttle_secs=1)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# tfrecord_train_files = glob.glob("nsfnet/tfrecords/train/*.tfrecords")
# tfrecord_eval_files = glob.glob("nsfnet/tfrecords/evaluate/*.tfrecords")

tfrecord_train_files = glob.glob("gbn/tfrecords/train/*.tfrecords")
tfrecord_eval_files = glob.glob("gbn/tfrecords/evaluate/*.tfrecords")

@dataclass
class HyperParams:
    # node_count: int = 14 #nsfnet
    node_count: int = 17 #gbn
    link_state_dim: int = 16
    path_state_dim: int = 32
    T: int = 8
    readout_units: int = 256
    learning_rate: float = 0.001
    batch_size: int = 32
    dropout_rate: float = 0.5
    l2: float = 0.01

@dataclass
class Args:
    target: str = "delay"
    strategy: Optional[str] = tf.distribute.MirroredStrategy() #None
    hparams: HyperParams = HyperParams()
    train: list = field(default_factory=lambda: tfrecord_train_files)
    eval_: list = field(default_factory=lambda: tfrecord_eval_files)
    model_dir: str = "LWO"
    train_steps: int = 50000
    eval_steps: Optional[int] = None
    shuffle_buf: int = 30000
    warm: Optional[str] = None

args = Args()


if __name__ == "__main__":
    train(args)
