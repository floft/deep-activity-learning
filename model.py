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
def flip_gradient(x, grl_lambda=1.0):
    """ Forward pass identity, backward pass negate gradient and multiply by l """
    grl_lambda = tf.cast(grl_lambda, dtype=tf.float32)
    zero = tf.constant(0, dtype=tf.float32)

    def grad(dy):
        # Fix the Num gradients 2 generated for op name ... do not match num
        # inputs 3 error. Though, we really don't care about the gradient of
        # of grl_lambda.
        return (tf.negative(dy) * grl_lambda, zero)

    return x, grad

class FlipGradient(tf.keras.layers.Layer):
    """ Gradient reversal layer """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @tf.function
    def call(self, inputs, grl_lambda=1.0, training=None):
        return flip_gradient(inputs, grl_lambda)

class StopGradient(tf.keras.layers.Layer):
    """ Stop gradient layer """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @tf.function
    def call(self, inputs, training=None):
        return tf.stop_gradient(inputs)

class ResnetFeatureExtractor(tf.keras.Model):
    """ Feature extractor with residual connections """
    def __init__(self, layers, units, dropout):
        super().__init__()

        self.seqs = []
        self.adds = []

        for _ in range(layers):
            self.seqs.append(tf.keras.Sequential([
                tf.keras.layers.Dense(units),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dropout(dropout),

                tf.keras.layers.Dense(units),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dropout(dropout),
            ]))

            self.adds.append(tf.keras.layers.Add())

    def call(self, inputs, training=None):
        net = inputs

        for i, (seq, add) in enumerate(zip(self.seqs, self.adds)):
            shortcut = net
            net = seq(net)

            if i > 0:
                net = add([shortcut, net])

        return net

def classifier(layers, units, dropout, num_classes, name=None):
    """ MLP classifier """
    model = tf.keras.Sequential()

    for _ in range(layers):
        model.add(tf.keras.layers.Dense(units))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation("softmax", name=name))
    return model

class DomainClassifier(tf.keras.Model):
    """ classifier() but stopping/flipping gradients and concatenating if generalization """
    def __init__(self, layers, units, dropout, num_domains, name=None):
        super().__init__()
        self.stop_gradient = StopGradient()
        self.flip_gradient = FlipGradient()
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.classifier = classifier(layers, units, dropout, num_domains, name)

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

class BaseModel(tf.keras.Model):
    """ Base model that we'll derive variations from """
    def __init__(self, num_classes, num_domains):
        super().__init__()

        # Feature extractor
        self.fe = ResnetFeatureExtractor(FLAGS.layers, FLAGS.units, FLAGS.dropout)

        # Task classifier
        self.task_classifier = classifier(FLAGS.task_layers, FLAGS.units,
            FLAGS.dropout, num_classes, name="task_output")

        # Domain classifier
        self.domain_classifier = DomainClassifier(FLAGS.domain_layers, FLAGS.units,
            FLAGS.dropout, num_domains, name="domain_output")

    def call(self, inputs, grl_lambda=1.0, training=None):
        fe = self.fe(inputs, training=training)
        task = self.task_classifier(fe, training=training)
        domain = self.domain_classifier([fe, task], grl_lambda=grl_lambda, training=training)
        return task, domain

class FlatModel(BaseModel):
    """ Flatten and normalize """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self, inputs, grl_lambda=1.0, training=None):
        net = self.pre(inputs, training=training)
        return super().call(net, grl_lambda, training)

@tf.function
def task_loss(y_true, y_pred, training):
    """
    Compute loss on the outputs of the task classifier

    Note: domain classifier can use normal tf.keras.losses.CategoricalCrossentropy
    but for the task loss when doing adaptation we need to ignore the second half
    of the batch since this is unsupervised
    """
    # Basically the same but only on half the batch
    cce = tf.keras.losses.CategoricalCrossentropy()

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

@tf.function
def domain_loss(y_true, y_pred, training):
    """ Compute loss on the outputs of the domain classifier """
    cce = tf.keras.losses.CategoricalCrossentropy()
    return cce(y_true, y_pred)

@tf.function
def compute_accuracy(y_true, y_pred):
    return tf.reduce_mean(input_tensor=tf.cast(tf.equal(
            tf.argmax(y_true, axis=-1),
            tf.argmax(y_pred, axis=-1)),
        tf.float32))

def make_model(num_classes, num_domains):
    """ Make the model """
    if FLAGS.model == "flat":
        model = FlatModel(num_classes, num_domains)

    return model
