"""
Create models

Primarily we can't be using just sequential because of the grl_lambda needing
to be passed around. Also residual connections need to be a separate custom
layer.
"""
import numpy as np
import tensorflow as tf

from absl import flags
from tensorflow.python.keras import backend as K

FLAGS = flags.FLAGS

flags.DEFINE_enum("model", "flat", ["flat"], "What model type to use")
flags.DEFINE_float("dropout", 0.05, "Dropout probability")
flags.DEFINE_integer("units", 50, "Number of LSTM hidden units and VRNN latent variable size")
flags.DEFINE_integer("layers", 5, "Number of layers for the feature extractor")
flags.DEFINE_integer("task_layers", 1, "Number of layers for the task classifier")
flags.DEFINE_integer("domain_layers", 2, "Number of layers for the domain classifier")

#flags.mark_flag_as_required("model")
flags.register_validator("dropout", lambda v: v != 0, message="dropout cannot be zero")

def make_flip_gradient():
    """ Only create constant once """
    zero = tf.constant(0, dtype=tf.float32)

    @tf.custom_gradient
    def flip_gradient(x, grl_lambda=1.0):
        """ Forward pass identity, backward pass negate gradient and multiply by l """
        grl_lambda = tf.cast(grl_lambda, dtype=tf.float32)

        def grad(dy):
            # Fix the "Num gradients 2 generated for op name ... do not match num
            # inputs 3" error. Though, we really don't care about the gradient of
            # of grl_lambda.
            return (tf.negative(dy) * grl_lambda, zero)

        return x, grad

    return flip_gradient

class FlipGradient(tf.keras.layers.Layer):
    """ Gradient reversal layer """
    def __init__(self):
        super().__init__()
        self.flip_gradient = make_flip_gradient()

    def call(self, inputs, grl_lambda=1.0, training=None):
        return self.flip_gradient(inputs, grl_lambda)

class StopGradient(tf.keras.layers.Layer):
    """ Stop gradient layer """
    @tf.function
    def call(self, inputs, training=None):
        return tf.stop_gradient(inputs)

class DenseBlock(tf.keras.layers.Layer):
    """
    Dense block with batch norm and dropout

    Note: doing this primarily because dense gives error if we pass
    training=True to it
    """
    def __init__(self, units, dropout):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation("relu")
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None):
        net = self.dense(inputs)
        net = self.bn(net, training=training)
        net = self.act(net)
        net = self.dropout(net, training=training)
        return net

class ResnetBlock(tf.keras.layers.Layer):
    """ Block consisting of other blocks but with residual connections """
    def __init__(self, units, dropout, layers=2, make_block=DenseBlock):
        super().__init__()
        self.blocks = [make_block(units, dropout)]*layers
        self.add = tf.keras.layers.Add()

    def call(self, inputs, training=None):
        """ Like Sequential but with a residual connection """
        shortcut = inputs
        net = inputs

        for block in self.blocks:
            net = block(net, training=training)

        return self.add([shortcut, net])

class Classifier(tf.keras.layers.Layer):
    """ MLP classifier -- multiple DenseBlock followed by dense of size num_classes and softmax """
    def __init__(self, layers, units, dropout, num_classes, make_block=DenseBlock):
        super().__init__()
        self.blocks = [make_block(units, dropout)]*layers
        self.dense = tf.keras.layers.Dense(num_classes)
        self.act = tf.keras.layers.Activation("softmax")

    def call(self, inputs, training=None):
        net = inputs

        for block in self.blocks:
            net = block(net, training=training)

        net = self.dense(net)
        net = self.act(net)

        return net

class DomainClassifier(tf.keras.layers.Layer):
    """ Classifier() but stopping/flipping gradients and concatenating if generalization """
    def __init__(self, layers, units, dropout, num_domains):
        super().__init__()
        self.stop_gradient = StopGradient()
        self.flip_gradient = FlipGradient()
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.classifier = Classifier(layers, units, dropout, num_domains)

    def call(self, inputs, grl_lambda=1.0, training=None):
        assert len(inputs) == 2, "Must call DomainClassiferModel(...)([fe, task])"
        net = inputs[0] # feature extractor output
        task = inputs[1] # task classifier output

        # For generalization, we also pass in the task classifier output to the
        # discriminator though forbid gradients from being passed through to
        # the task classifier.
        if FLAGS.generalize:
            task_stop_gradient = self.stop_gradient(task)
            net = self.concat([net, task_stop_gradient])

        net = self.flip_gradient(net, grl_lambda=grl_lambda, training=training)
        net = self.classifier(net, training=training)
        return net

class FeatureExtractor(tf.keras.layers.Layer):
    """ Resnet feature extractor """
    def __init__(self, layers, units, dropout):
        super().__init__()
        # First can't be residual since x isn't of size units
        self.blocks = [DenseBlock(units, dropout)]
        self.blocks += [ResnetBlock(units, dropout)]*layers

    def call(self, inputs, training=None):
        net = inputs

        for block in self.blocks:
            net = block(net, training=training)

        return net

class FlatModel(tf.keras.layers.Layer):
    """ Flatten and normalize then model """
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        net = self.flatten(inputs)
        return self.bn(net, training=training)

class DomainAdaptationModel(tf.keras.Model):
    """
    Contains custom model, feature extractor, task classifier, and domain
    classifier

    The custom model before the feature extractor depends on the command line
    argument.

    Usage:
        model = DomainAdaptationModel(num_classes, num_domains)

        with tf.GradientTape() as tape:
            task_y_pred, domain_y_pred = model(x, grl_lambda=1.0, training=True)
            ...
    """
    def __init__(self, num_classes, num_domains):
        super().__init__()
        self.feature_extractor = FeatureExtractor(FLAGS.layers, FLAGS.units, FLAGS.dropout)
        self.task_classifier = Classifier(FLAGS.task_layers, FLAGS.units,
            FLAGS.dropout, num_classes)
        self.domain_classifier = DomainClassifier(FLAGS.domain_layers, FLAGS.units,
            FLAGS.dropout, num_domains)

        if FLAGS.model == "flat":
            self.custom_model = FlatModel()

    def call(self, inputs, grl_lambda=1.0, training=None):
        net = self.custom_model(inputs, training=training)
        net = self.feature_extractor(net, training=training)
        task = self.task_classifier(net, training=training)
        domain = self.domain_classifier([net, task], grl_lambda=grl_lambda, training=training)
        return task, domain

def make_task_loss():
    """
    The same as CategoricalCrossentropy() but only on half the batch if doing
    adaptation and in the training phase
    """
    cce = tf.keras.losses.CategoricalCrossentropy()

    @tf.function
    def task_loss(y_true, y_pred, training=None):
        """
        Compute loss on the outputs of the task classifier

        Note: domain classifier can use normal tf.keras.losses.CategoricalCrossentropy
        but for the task loss when doing adaptation we need to ignore the second half
        of the batch since this is unsupervised
        """
        if training is None:
            training = K.learning_phase()

        # If doing domain adaptation, then we'll need to ignore the second half of the
        # batch for task classification during training since we don't know the labels
        # of the target data
        if FLAGS.adapt and training:
            # Note: this is twice the batch_size in the train() function since we cut
            # it in half there -- this is the sum of both source and target data
            batch_size = tf.shape(y_pred)[0]

            y_pred = tf.slice(y_pred, [0, 0], [batch_size // 2, -1])
            y_true = tf.slice(y_true, [0, 0], [batch_size // 2, -1])

        return cce(y_true, y_pred)

    return task_loss

def make_domain_loss():
    """
    Just CategoricalCrossentropy() but for consistency with make_task_loss()
    """
    cce = tf.keras.losses.CategoricalCrossentropy()

    @tf.function
    def domain_loss(y_true, y_pred):
        """ Compute loss on the outputs of the domain classifier """
        return cce(y_true, y_pred)

    return domain_loss

@tf.function
def compute_accuracy(y_true, y_pred):
    return tf.reduce_mean(input_tensor=tf.cast(tf.equal(
            tf.argmax(y_true, axis=-1),
            tf.argmax(y_pred, axis=-1)),
        tf.float32))
