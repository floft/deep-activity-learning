"""
Create models
"""
import numpy as np
import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_enum("model", None, ["flat"], "What model type to use")
flags.DEFINE_float("dropout", 0.05, "Dropout probability")
flags.DEFINE_integer("units", 50, "Number of LSTM hidden units and VRNN latent variable size")
flags.DEFINE_integer("layers", 5, "Number of layers for the feature extractor")
flags.DEFINE_integer("task_layers", 1, "Number of layers for the task classifier")
flags.DEFINE_integer("domain_layers", 2, "Number of layers for the domain classifier")

flags.mark_flag_as_required("model")
flags.register_validator("dropout", lambda v: v != 0, message="dropout cannot be zero")

@tf.custom_gradient
def flip_gradient(x, l=1.0):
    """ Forward pass identity, backward pass negate gradient and multiply by l """
    def grad(dy):
        return tf.negative(dy) * l
    return x, grad

class FlipGradient(tf.keras.layers.Layer):
    """ Gradient reversal layer """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @tf.function
    def call(self, inputs, l=1.0, training=None):
        return flip_gradient(inputs, l)

class StopGradient(tf.keras.layers.Layer):
    """ Stop gradient layer """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @tf.function
    def call(self, inputs, training=None):
        return tf.stop_gradient(inputs)

def feature_extractor(net, layers, units, dropout):
    """ Feature extractor with residual connections """
    for i in range(layers):
        shortcut = net

        net = tf.keras.layers.Dense(units, activation="uniform")(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)
        net = tf.keras.layers.Dropout(dropout)(net)

        net = tf.keras.layers.Dense(units, activation="uniform")(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)
        net = tf.keras.layers.Dropout(dropout)(net)

        if i > 0:
            net = tf.keras.layers.Add()([shortcut, net])

    return net

def classifier(net, layers, units, dropout, num_classes, name=None):
    """ MLP classifier """
    for _ in range(layers):
        net = tf.keras.layers.Dense(units, activation="uniform")(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)
        net = tf.keras.layers.Dropout(dropout)(net)

    net = tf.keras.layers.Dense(num_classes, activation="uniform")(net)
    net = tf.keras.layers.Activation("softmax", name=name)(net)
    return net

def make_flat(input_shape, num_classes, num_domains, grl_lambda):
    """ Flatten the input and pass directly to the feature extractor """
    inputs = tf.keras.layers.Input(input_shape)

    # Flatten and normalize
    net = tf.keras.layers.Flatten()(inputs)
    net = tf.keras.layers.BatchNormalization()(net) # i.e. normalize inputs...

    # Feature extractor
    net = feature_extractor(net, FLAGS.layers, FLAGS.units, FLAGS.dropout)

    # Task classifier
    task = classifier(net, FLAGS.task_layers, FLAGS.units, FLAGS.dropout,
        num_classes, name="task_output")

    # Domain classifier
    #
    # For generalization, we also pass in the task classifier output to the
    # discriminator though forbid gradients from being passed through to
    # the task classifier.
    if FLAGS.generalization:
        task_stop_gradient = StopGradient()(task)
        domain = tf.keras.layers.Concatenate()([net, task_stop_gradient])
    else:
        domain = net

    domain = FlipGradient()(domain)
    domain = classifier(domain, FLAGS.domain_layers, FLAGS.units, FLAGS.dropout,
        num_domains, name="domain_output")

    return tf.keras.Model(inputs=inputs, outputs=[task,domain])

def make_task_loss(training):
    """
    Compute loss on the outputs of the task classifier

    Note: domain classifier can use normal tf.keras.losses.CategoricalCrossentropy
    but for the task loss when doing adaptation we need to ignore the second half
    of the batch since this is unsupervised
    """
    def loss(y_true, y_pred):
        # Basically the same but only on half the batch
        cce = tf.keras.losses.CategoricalCrossentropy()

        # If doing domain adaptation, then we'll need to ignore the second half of the
        # batch for task classification during training since we don't know the labels
        # of the target data
        if FLAGS.adaptation and training:
            # Note: this is twice the batch_size in the train() function since we cut
            # it in half there -- this is the sum of both source and target data
            batch_size = tf.shape(y_pred)[0]

            y_pred = tf.slice(y_pred, [0, 0], [batch_size // 2, -1])
            y_true = tf.slice(y_true, [0, 0], [batch_size // 2, -1])

        return cce(y_pred, y_true)

    return loss

def make_model(x_dims, num_classes, num_domains, grl_lambda, lr, training):
    """ Make the model, create the losses, metrics, etc. """

    if FLAGS.model == "flat":
        model = make_flat(x_dims, num_classes, num_domains, grl_lambda)

    loss = {"task_output": make_task_loss(training)}
    loss_weights = {"task_output": 1.0}

    if FLAGS.adaptation or FLAGS.generalization:
        loss["domain_output"] = tf.keras.losses.CategoricalCrossentropy()
        loss_weights["domain_output"] = 1.0

    opt = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights,
        metrics=["accuracy"])
