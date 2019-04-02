"""
Update metrics for displaying in TensorBoard
"""
import time
import tensorflow as tf

from absl import flags

from load_data import domain_labels
from model import task_loss, domain_loss

FLAGS = flags.FLAGS
flags.DEFINE_integer("max_examples", 0, "Max number of examples to evaluate for validation (default 0, i.e. all)")

def run_multi_batch(model, data, domain, num_domains, after_batch, max_examples=0):
    """
    Evaluate model on all the data up to max_examples, even if it takes multiple
    batches. Calls:
        after_batch([labels_batch_a, task_y_pred, domains_batch_a, domain_y_pred,
            total_loss, task_loss, domain_loss])

    after each batch is completed. Set max_examples==0 for evaluating on all data.
    domain should be either 0 or 1 (if num_domains==2)
    """
    examples = 0

    if data is None:
        return

    for x, task_y_true, domain_y_true in data:
        # Make sure we don't go over the desired number of examples
        # But, only if we don't want to evaluate all examples (i.e. if
        # max_examples == 0)
        if max_examples != 0 and examples >= max_examples:
            break

        if max_examples != 0:
            diff = max_examples - examples

            if examples + x.shape[0] > max_examples:
                x = x[:diff]
                task_y_true = task_y_true[:diff]

        batch_size = x.shape[0]
        examples += batch_size

        # Match the number of examples we have since the domain_y_true
        # is meant for generalization, where it's a different number for
        # each home
        if not FLAGS.generalize:
            domain_y_true = domain_labels(domain, batch_size, num_domains)

        # Evaluate model on data
        task_y_pred, domain_y_pred = model(x, training=False)

        # Calculate losses
        task_l = task_loss(task_y_true, task_y_pred, training=False)
        domain_l = domain_loss(domain_y_true, domain_y_pred, training=False)
        total_l = task_l + domain_l

        # Do whatever they want with the results of this batch
        after_batch([
            task_y_true, task_y_pred, domain_y_true, domain_y_pred,
            total_l, task_l, domain_l,
        ])

class Metrics:
    """
    Handles keeping track of metrics either over one batch or many batch, then
    after all (or just the one) batches are processed, saving this to a log file
    for viewing in TensorBoard.

    Accuracy values:
        accuracy_{domain,task}/{source,target}/{training,validation}
        accuracy_task_class_{class1name,...}/{source,target}/{training,validation}

    Loss values:
        loss/{total,task,domain}
    """
    def __init__(self, writer, num_classes, num_domains, config):
        self.writer = writer
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.config = config
        self.datasets = ["training", "validation"]

        if FLAGS.generalize:
            self.domains = ["source"]
        else:
            self.domains = ["source", "target"]

        # Create all entire-batch categorical accuracy metrics
        self.categorical_metrics = {}
        for domain in self.domains:
            for dataset in self.datasets:
                for name in ["domain", "task"]:
                    n = "accuracy_%s/%s/%s"%(name, domain, dataset)
                    self.categorical_metrics[n] = tf.keras.metrics.CategoricalAccuracy(name=n)

        # Create all per-class accuracy metrics
        self.per_class_metrics = {}
        for i in range(self.num_classes):
            class_name = self.config.int_to_label(i)

            for domain in self.domains:
                for dataset in self.datasets:
                    n = "accuracy_task_class_%s/%s/%s"%(class_name, domain, dataset)
                    self.per_class_metrics[n] = tf.keras.metrics.Accuracy(name=n)

        # Losses
        self.loss_total = tf.keras.metrics.Mean(name="loss/total")
        self.loss_task = tf.keras.metrics.Mean(name="loss/task")
        self.loss_domain = tf.keras.metrics.Mean(name="loss/domain")

    def _reset_states(self):
        """ Reset states of all the Keras metrics """
        for _, v in self.categorical_metrics.items():
            v.reset_states()

        for _, v in self.per_class_metrics.items():
            v.reset_states()

        self.loss_total.reset_states()
        self.loss_task.reset_states()
        self.loss_domain.reset_states()

    def _process_losses(self, results):
        """ Update loss values """
        _, _, _, _, \
            total_loss, task_loss, domain_loss = results
        self.loss_total(total_loss)
        self.loss_task(task_loss)
        self.loss_domain(domain_loss)

    def _process_accuracy(self, results, domain, dataset):
        """ Update metrics for accuracy over entire batch for domain-dataset """
        task_y_true, task_y_pred, domain_y_true, domain_y_pred, \
            _, _, _ = results

        # Update domain accuracy
        n = "accuracy_domain/%s/%s"%(domain, dataset)
        self.categorical_metrics[n](domain_y_true, domain_y_pred)

        # Update task accuracy
        n = "accuracy_task/%s/%s"%(domain, dataset)
        self.categorical_metrics[n](task_y_true, task_y_pred)

    def _process_accuracy_per_class(self, results, domain, dataset):
        """ Update metrics for accuracy over per-class portions of batch for domain-dataset """
        task_y_true, task_y_pred, _, _, _, _, _ = results
        batch_size = tf.shape(task_y_true)[0]

        # If only predicting a single class (using softmax), then look for the
        # max value
        # e.g. [0.2 0.2 0.4 0.2] -> [0 0 1 0]
        per_class_predictions = tf.one_hot(
            tf.argmax(task_y_pred, axis=-1), self.num_classes)

        for i in range(self.num_classes):
            class_name = self.config.int_to_label(i)

            # Get ith column (all groundtruth/predictions for ith class)
            y_true = tf.slice(task_y_true, [0,i], [batch_size,1])
            y_pred = tf.slice(per_class_predictions, [0,i], [batch_size,1])

            # For single-class prediction, we want to first isolate which
            # examples in the batch were supposed to be class X. Then, of
            # those, calculate accuracy = correct / total.
            rows_of_class_y = tf.where(tf.equal(y_true, 1)) # i.e. have 1
            acc_y_true = tf.gather(y_true, rows_of_class_y)
            acc_y_pred = tf.gather(y_pred, rows_of_class_y)

            # Update task accuracy
            n = "accuracy_task_class_%s/%s/%s"%(class_name, domain, dataset)
            self.per_class_metrics[n](acc_y_true, acc_y_pred)

    def _write_data(self, step, dataset, eval_time, train_time=None):
        """ Write either the training or validation data """
        assert dataset in self.datasets, "unknown dataset "+str(dataset)

        # Write all the values to the file
        with self.writer.as_default():
            for domain in self.domains:
                for name in ["domain", "task"]:
                    n = "accuracy_%s/%s/%s"%(name, domain, dataset)
                    tf.summary.scalar(n, self.categorical_metrics[n].result(), step=step)

                for i in range(self.num_classes):
                    class_name = self.config.int_to_label(i)
                    n = "accuracy_task_class_%s/%s/%s"%(class_name, domain, dataset)
                    tf.summary.scalar(n, self.per_class_metrics[n].result(), step=step)

            tf.summary.scalar("step_time/metrics/%s"%(dataset), eval_time, step=step)

            if train_time is not None:
                tf.summary.scalar("step_time/%s"%(dataset), train_time, step=step)

            # Only log losses on training data
            if dataset == "training":
                tf.summary.scalar("loss/total", self.loss_total.result(), step=step)
                tf.summary.scalar("loss/task", self.loss_task.result(), step=step)
                tf.summary.scalar("loss/domain", self.loss_domain.result(), step=step)

        # Prepare for next time
        self._reset_states()

    def _process_partial(self, results, domain, dataset):
        """ Call this on each batch when running on train/test data (domain A = "source",
        then domain B = "target") to update the partial metric results """
        assert dataset in self.datasets, "unknown dataset "+str(dataset)
        assert domain in self.domains, "unknown domain "+str(domain)
        self._process_accuracy(results, domain, dataset)
        self._process_accuracy_per_class(results, domain, dataset)

        # Only log losses on training data
        if dataset == "training":
            self._process_losses(results)

    def _run_partial(self, model, data_a, data_b, dataset):
        """ Run the data A/B through the model """
        if FLAGS.generalize:
            run_multi_batch(model, data_a, None, None,
                lambda results: self._process_partial(results, "source", dataset),
                FLAGS.max_examples)
        else:
            run_multi_batch(model, data_a, 0, self.num_domains,
                lambda results: self._process_partial(results, "source", dataset),
                FLAGS.max_examples)
            run_multi_batch(model, data_b, 1, self.num_domains,
                lambda results: self._process_partial(results, "target", dataset),
                FLAGS.max_examples)

    def train(self, model, data_a, data_b, step, train_time):
        """ Call this once after evaluating on the training data for domain A
        and domain B """
        dataset = "training"
        step = int(step)

        # Only one batch is passed in for training, so make it a list so that
        # we can reuse the _run_partial function. However, only if we have data.
        if data_a is not None:
            data_a = [data_a]
        if data_b is not None:
            data_b = [data_b]

        t = time.time()
        self._run_partial(model, data_a, data_b, dataset)
        t = time.time() - t

        self._write_data(step, "training", t, train_time)

    def test(self, model, eval_data_a, eval_data_b, step):
        """ Evaluate the model on domain A/B but batched to make sure we don't
        run out of memory """
        dataset = "validation"
        step = int(step)

        t = time.time()
        self._run_partial(model, eval_data_a, eval_data_b, dataset)
        t = time.time() - t

        self._write_data(step, dataset, t)
