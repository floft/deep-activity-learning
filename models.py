"""
Models

Primarily we can't be using just sequential because of the grl_lambda needing
to be passed around. Also residual connections need to be a separate custom
layer.

Provides the model DomainAdaptationModel() and its components along with the
make_{task,domain}_loss() functions. Also, compute_accuracy() if desired.

Usage:
    # Build our model
    model = DomainAdaptationModel(num_classes, num_domains, "flat",
            generalize=False)

    # During training
    task_y_pred, domain_y_pred = model(x, grl_lambda=1.0, training=True)

    # During evaluation
    task_y_pred, domain_y_pred = model(x, training=False)
"""
import numpy as np
import tensorflow as tf

from absl import flags
from tensorflow.python.keras import backend as K

FLAGS = flags.FLAGS

flags.DEFINE_float("dropout", 0.05, "Dropout probability")
flags.DEFINE_integer("units", 50, "Number of LSTM hidden units and VRNN latent variable size")
flags.DEFINE_integer("layers", 5, "Number of layers for the feature extractor")
flags.DEFINE_integer("task_layers", 1, "Number of layers for the task classifier")
flags.DEFINE_integer("domain_layers", 2, "Number of layers for the domain classifier")
flags.DEFINE_integer("resnet_layers", 2, "Number of layers within a single resnet block")

flags.register_validator("dropout", lambda v: v != 1, message="dropout cannot be 1")

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flip_gradient = make_flip_gradient()

    def call(self, inputs, grl_lambda=1.0, training=None):
        return self.flip_gradient(inputs, grl_lambda)

class StopGradient(tf.keras.layers.Layer):
    """ Stop gradient layer """
    def call(self, inputs, training=None):
        return tf.stop_gradient(inputs)

class DenseBlock(tf.keras.layers.Layer):
    """
    Dense block with batch norm and dropout

    Note: doing this rather than Sequential because dense gives error if we pass
    training=True to it
    """
    def __init__(self, units, dropout, **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, units, dropout, layers=None,
            make_block=DenseBlock, **kwargs):
        super().__init__(**kwargs)

        if layers is None:
            layers = FLAGS.resnet_layers

        self.blocks = [make_block(units, dropout) for _ in range(layers)]
        self.add = tf.keras.layers.Add()

    def call(self, inputs, training=None):
        """ Like Sequential but with a residual connection """
        shortcut = inputs
        net = inputs

        for block in self.blocks:
            net = block(net, training=training)

        return self.add([shortcut, net])

class Classifier(tf.keras.layers.Layer):
    """ MLP classifier -- multiple DenseBlock followed by dense of size
    num_classes and softmax """
    def __init__(self, layers, units, dropout, num_classes,
            make_block=DenseBlock, **kwargs):
        super().__init__(**kwargs)
        assert layers > 0, "must have layers > 0"
        self.blocks = [make_block(units, dropout) for _ in range(layers-1)]
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
    """ Classifier() but stopping/flipping gradients and concatenating if
    generalization """
    def __init__(self, layers, units, dropout, num_domains, generalize, **kwargs):
        super().__init__(**kwargs)
        self.stop_gradient = StopGradient()
        self.flip_gradient = FlipGradient()
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.classifier = Classifier(layers, units, dropout, num_domains)
        self.generalize = generalize

    def call(self, inputs, grl_lambda=1.0, training=None):
        assert len(inputs) == 2, "Must call DomainClassiferModel(...)([fe, task])"
        net = inputs[0] # feature extractor output
        task = inputs[1] # task classifier output

        # For generalization, we also pass in the task classifier output to the
        # discriminator though forbid gradients from being passed through to
        # the task classifier.
        if self.generalize:
            task_stop_gradient = self.stop_gradient(task)
            net = self.concat([net, task_stop_gradient])

        net = self.flip_gradient(net, grl_lambda=grl_lambda, training=training)
        net = self.classifier(net, training=training)
        return net

class FeatureExtractor(tf.keras.layers.Layer):
    """ Resnet feature extractor """
    def __init__(self, layers, units, dropout,
            make_base_block=DenseBlock, make_res_block=ResnetBlock, **kwargs):
        super().__init__(**kwargs)
        assert layers > 0, "must have layers > 0"
        # First can't be residual since x isn't of size units
        self.blocks = [make_base_block(units, dropout) for _ in range(FLAGS.resnet_layers)]
        self.blocks += [make_res_block(units, dropout) for _ in range(layers-1)]

    def call(self, inputs, training=None):
        net = inputs

        for block in self.blocks:
            net = block(net, training=training)

        return net

class FlatModel(tf.keras.layers.Layer):
    """ Flatten and normalize then model """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.999)

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
        model = DomainAdaptationModel(num_classes, num_domains, "flat",
            generalize=False)

        with tf.GradientTape() as tape:
            task_y_pred, domain_y_pred = model(x, grl_lambda=1.0, training=True)
            ...
    """
    def __init__(self, num_classes, num_domains, model_name, generalize, **kwargs):
        super().__init__(**kwargs)

        self.feature_extractor = FeatureExtractor(FLAGS.layers, FLAGS.units, FLAGS.dropout)
        self.task_classifier = Classifier(FLAGS.task_layers, FLAGS.units,
            FLAGS.dropout, num_classes)
        self.domain_classifier = DomainClassifier(FLAGS.domain_layers, FLAGS.units,
            FLAGS.dropout, num_domains, generalize)

        if model_name == "flat":
            self.custom_model = FlatModel()
        else:
            raise NotImplementedError("Model name: "+str(model_name))

    @property
    def trainable_variables_exclude_domain(self):
        """ Same as .trainable_variables but excluding the domain classifier """
        return self.feature_extractor.trainable_variables \
            + self.task_classifier.trainable_variables \
            + self.custom_model.trainable_variables

    def call(self, inputs, grl_lambda=1.0, training=None):
        net = self.custom_model(inputs, training=training)
        net = self.feature_extractor(net, training=training)
        task = self.task_classifier(net, training=training)
        domain = self.domain_classifier([net, task], grl_lambda=grl_lambda, training=training)
        return task, domain

def make_task_loss(class_weights, adapt):
    """
    The same as CategoricalCrossentropy() but only on half the batch if doing
    adaptation and in the training phase
    """
    cce = tf.keras.losses.CategoricalCrossentropy()

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
        if adapt and training:
            # Note: this is twice the batch_size in the train() function since we cut
            # it in half there -- this is the sum of both source and target data
            batch_size = tf.shape(y_pred)[0]

            y_pred = tf.slice(y_pred, [0, 0], [batch_size // 2, -1])
            y_true = tf.slice(y_true, [0, 0], [batch_size // 2, -1])

        # Tile the class weights to match the batch size
        #
        # e.g., if the weights are [1,2,3,4] and we have a batch of size 2, we get:
        #  [[1,2,3,4],
        #   [1,2,3,4]]
        if not isinstance(class_weights, float) and not isinstance(class_weights, int):
            # There needs to be one weight for each item in the batch based on
            # which class that item was predicted to be
            #
            # e.g. if the classes should be [[0,1],[1,0],[1,0]] (i.e. class 1,
            # class 0, class 0) for a batch size of two, and we have class weights
            # [2,3] we should output: [3,2,2] for the weights for this batch
            which_label = tf.argmax(y_true, axis=-1) # e.g. [1,0,0] for above
            # Then, get the weights based on which class each was
            batch_class_weights = tf.gather(class_weights, which_label)
        # If it's just the default 1.0 or some scalar, then don't bother
        # expanding to match the batch size
        else:
            batch_class_weights = class_weights

        return cce(y_true, y_pred, sample_weight=batch_class_weights)

    return task_loss

def make_domain_loss(use_domain_loss):
    """
    Just CategoricalCrossentropy() but for consistency with make_task_loss()
    """
    if use_domain_loss:
        cce = tf.keras.losses.CategoricalCrossentropy()

        def domain_loss(y_true, y_pred):
            """ Compute loss on the outputs of the domain classifier """
            return cce(y_true, y_pred)
    else:
        def domain_loss(y_true, y_pred):
            """ Domain loss only used during adaptation or generalization """
            return 0

    return domain_loss

def compute_accuracy(y_true, y_pred):
    return tf.reduce_mean(input_tensor=tf.cast(tf.equal(
            tf.argmax(y_true, axis=-1),
            tf.argmax(y_pred, axis=-1)),
        tf.float32))
