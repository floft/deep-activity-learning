"""
Create models

This provides the functions like build_lstm and build_vrnn that are used in training.
"""
import numpy as np
import tensorflow as tf
layers = tf.contrib.layers
framework = tf.contrib.framework

from flip_gradient import flip_gradient

def classifier(x, num_classes, keep_prob, training, batch_norm, units=50, num_layers=1):
    """
    We'll use the same clasifier for task or domain classification

    Same as was used in the VRADA paper (see paper appendix)

    Returns both output without applying softmax for use in loss function
    and after for use in prediction. See softmax_cross_entropy_with_logits_v2
    documentation: "This op expects unscaled logits, ..."

    Also returns sigmoid output for if doing multi-class classification.
    """
    classifier_output = x

    for i in range(num_layers):
        with tf.compat.v1.variable_scope("layer_"+str(i)):
            # Last layer has desired output size, otherwise use a fixed size
            if i == num_layers-1:
                num_features = num_classes
            else:
                num_features = units

            classifier_output = tf.contrib.layers.fully_connected(
                    classifier_output, num_features, activation_fn=None)

            # Last activation is softmax, which we will apply afterwards
            if i != num_layers-1:
                if batch_norm:
                    classifier_output = tf.compat.v1.layers.batch_normalization(
                        classifier_output, training=training)

                classifier_output = tf.nn.relu(classifier_output)
                classifier_output = tf.nn.dropout(classifier_output, 1 - (keep_prob))

    sigmoid_output = tf.nn.sigmoid(classifier_output)
    softmax_output = tf.nn.softmax(classifier_output)

    return classifier_output, softmax_output, sigmoid_output

def build_model(x, y, domain, grl_lambda, keep_prob, training,
        num_classes, num_domains, adaptation=True, generalization=False,
        multi_class=False, class_weights=1.0, units=50, layers=5,
        batch_norm=True, log_outputs=False,
        use_grl=True, use_feature_extractor=True):
    """
    Creates the feature extractor, task classifier, domain classifier

    Inputs:
        x -- fed into feature extractor
        y -- task labels
        domain -- [[1,0], [0,1], ...] for source or target domain
        glr_lambda -- float placeholder for lambda for gradient reversal layer
        keep_prob -- float placeholder for dropout probability
        training -- boolean placeholder for if we're training
        adaptation -- boolean whether we wish to perform adaptation or not
        multi_class -- boolean whether to use sigmoid (for multi-class) or softmax
        batch_norm -- boolean whether to use BatchNorm
        log_outputs -- boolean whether we want to log outputs to for TensorBoard
        class_weights -- weights for handling large class imbalances (probably
            pass in [class0_weight, class1_weight, ... classN_weight])
    Outputs:
        task_output, domain_softmax -- predictions of classifiers
        task_loss, domain_loss -- losses
        feature_extractor -- output of feature extractor (e.g. for t-SNE)
        summaries -- more summaries to save
    """

    with tf.compat.v1.variable_scope("feature_extractor"):
        feature_extractor = x
        num_layers = 0

        if use_feature_extractor:
            num_layers = layers

        for i in range(num_layers):
            with tf.compat.v1.variable_scope("layer_"+str(i)):
                n = feature_extractor

                n = tf.contrib.layers.fully_connected(
                    n, units, activation_fn=None)
                if batch_norm:
                    n = tf.compat.v1.layers.batch_normalization(
                        n, training=training)
                n = tf.nn.relu(n)
                n = tf.nn.dropout(n, 1 - (keep_prob))

                n = tf.contrib.layers.fully_connected(
                    n, units, activation_fn=None)
                if batch_norm:
                    n = tf.compat.v1.layers.batch_normalization(
                        n, training=training)
                n = tf.nn.relu(n)
                n = tf.nn.dropout(n, 1 - (keep_prob))

                # Make this kind of like residual networks, where the new layer
                # learns the change from the previous value, i.e. we do previous
                # layer output plus the new layer's output. Assuming they have
                # the same dimension, which is the case for all but the first
                # layer
                if i == 0:
                    feature_extractor = n
                else:
                    feature_extractor = feature_extractor + n

    # Pass last output to fully connected then softmax to get class prediction
    with tf.compat.v1.variable_scope("task_classifier"):
        task_classifier, task_softmax, task_sigmoid = classifier(
            feature_extractor, num_classes, keep_prob, training, batch_norm,
            units, num_layers=1)

    # Also pass output to domain classifier
    # Note: always have 2 domains, so set outputs to 2
    with tf.compat.v1.variable_scope("domain_classifier"):
        # Optionally bypass using a GRL
        if use_grl:
            gradient_reversal_layer = flip_gradient(feature_extractor, grl_lambda)
        else:
            gradient_reversal_layer = feature_extractor

        # For generalization, we also pass in the task classifier output to the
        # discriminator though forbid gradients from being passed through to
        # the task classifier.
        if generalization:
            gradient_reversal_layer = tf.concat([
                tf.stop_gradient(task_softmax), gradient_reversal_layer], axis=1)

        domain_classifier, domain_softmax, _ = classifier(
            gradient_reversal_layer, num_domains, keep_prob, training, batch_norm,
            units, num_layers=2)

    # If doing domain adaptation, then we'll need to ignore the second half of the
    # batch for task classification during training since we don't know the labels
    # of the target data
    if adaptation:
        with tf.compat.v1.variable_scope("only_use_source_labels"):
            # Note: this is twice the batch_size in the train() function since we cut
            # it in half there -- this is the sum of both source and target data
            batch_size = tf.shape(input=feature_extractor)[0]

            # Note: I'm doing this after the classification layers because if you do
            # it before, then fully_connected complains that the last dimension is
            # None (i.e. not known till we run the graph). Thus, we'll do it after
            # all the fully-connected layers.
            #
            # Alternatively, I could do matmul(weights, task_input) + bias and store
            # weights on my own if I do really need to do this at some point.
            #
            # See: https://github.com/pumpikano/tf-dann/blob/master/Blobs-DANN.ipynb
            task_classifier = tf.cond(pred=training,
                true_fn=lambda: tf.slice(task_classifier, [0, 0], [batch_size // 2, -1]),
                false_fn=lambda: task_classifier)
            task_softmax = tf.cond(pred=training,
                true_fn=lambda: tf.slice(task_softmax, [0, 0], [batch_size // 2, -1]),
                false_fn=lambda: task_softmax)
            task_sigmoid = tf.cond(pred=training,
                true_fn=lambda: tf.slice(task_sigmoid, [0, 0], [batch_size // 2, -1]),
                false_fn=lambda: task_sigmoid)
            y = tf.cond(pred=training,
                true_fn=lambda: tf.slice(y, [0, 0], [batch_size // 2, -1]),
                false_fn=lambda: y)

    # Losses
    with tf.compat.v1.variable_scope("task_loss"):
        # Tile the class weights to match the batch size
        #
        # e.g., if the weights are [1,2,3,4] and we have a batch of size 2, we get:
        #  [[1,2,3,4],
        #   [1,2,3,4]]
        if not isinstance(class_weights, float) and not isinstance(class_weights, int):
            class_weights_reshape = tf.reshape(class_weights,
                [1,tf.shape(input=class_weights)[0]])
            tiled_class_weights = tf.tile(class_weights_reshape,
                [tf.shape(input=y)[0],1])

            # If not multi-class, then there needs to be one weight for each
            # item in the batch based on which class that item was predicted to
            # be
            #
            # e.g. if we predicted classes [[0,1],[1,0],[1,0]] (i.e. class 1,
            # class 0, class 0) for a batch size of two, and we have weights
            # [2,3] we should output: [3,2,2] for the weights for this batch
            which_label = tf.argmax(input=task_classifier, axis=-1) # e.g. [1,0,0] for above
            # Then, get the weights based on which class each was
            batch_class_weights = tf.gather(class_weights, which_label)
        # If it's just the default 1.0 or some scalar, then don't bother
        # expanding to match the batch size
        else:
            tiled_class_weights = class_weights
            batch_class_weights = class_weights

        # If multi-class (i.e. predict any number of the classes not necessarily
        # just one), use a different TensorFlow loss function that treats each
        # output separately (not doing softmax, where we care about the max one)
        if multi_class:
            task_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
                y, task_classifier, tiled_class_weights)
        else:
            task_loss = tf.compat.v1.losses.softmax_cross_entropy(
                y, task_classifier, batch_class_weights)

    with tf.compat.v1.variable_scope("domain_loss"):
        domain_loss = tf.compat.v1.losses.softmax_cross_entropy(domain, domain_classifier)

    # If multi-class the task output will be sigmoid rather than softmax
    if multi_class:
        task_output = task_sigmoid
    else:
        task_output = task_softmax

    # Extra summaries
    summaries = [
        tf.compat.v1.summary.scalar("loss/task_loss", task_loss),
        tf.compat.v1.summary.scalar("loss/domain_loss", domain_loss),
    ]

    if log_outputs:
        summaries += [
            tf.compat.v1.summary.histogram("outputs/feature_extractor", feature_extractor),
            tf.compat.v1.summary.histogram("outputs/domain_classifier", domain_softmax),
        ]

        with tf.compat.v1.variable_scope("outputs"):
            for i in range(num_classes):
                summaries += [
                    tf.compat.v1.summary.histogram("task_classifier_%d" % i,
                        tf.slice(task_output, [0,i], [tf.shape(input=task_output)[0],1]))
                ]

    return task_output, domain_softmax, task_loss, domain_loss, \
        feature_extractor, summaries

def build_flat(x, y, domain, grl_lambda, keep_prob, training,
            num_classes, num_domains, num_features, adaptation, generalization, units, layers,
            multi_class=False, bidirectional=False, class_weights=1.0,
            x_dims=None, use_feature_extractor=True, initial_batch_norm=True):
    """ Flatten the input and pass directly to the feature extractor

    Note: only need x_dims in build_flat none of the other build_* since here
    we must know the time steps * features since we directly feed into a dense
    layer, which requires a known size.
    """
    # Only flatten data -- reshape from [batch, time steps, features]
    # to be [batch, time steps * features], i.e. [batch_size, -1] except Dense
    # doesn't work with size None
    with tf.compat.v1.variable_scope("flat_model"):
        output = tf.reshape(x, [tf.shape(input=x)[0], np.prod(x_dims)])

        if initial_batch_norm:
            output = tf.compat.v1.layers.batch_normalization(output, momentum=0.999, training=training)

    # Other model components passing in output from above
    task_output, domain_softmax, task_loss, domain_loss, \
        feature_extractor, summaries = build_model(
            output, y, domain, grl_lambda, keep_prob, training,
            num_classes, num_domains, adaptation, generalization, multi_class, class_weights,
            units, layers, use_feature_extractor=use_feature_extractor)

    # Total loss is the sum
    with tf.compat.v1.variable_scope("total_loss"):
        total_loss = task_loss

        if adaptation or generalization:
            total_loss += domain_loss

    extra_outputs = None

    return task_output, domain_softmax, total_loss, \
        feature_extractor, summaries, extra_outputs
