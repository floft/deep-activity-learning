"""
Metrics

Update metrics for displaying in TensorBoard during training or evaluation after
training

Usage during training (logging to a log file for TensorBoard):
    metrics = Metrics(log_dir, num_classes, num_domains, al_config,
        task_loss_fn, domain_loss_fn, generalize, domain_b_data is not None)

    # Evaluate model on a single training batch, update metrics, save to log file
    metrics.train(model, train_data_a, train_data_b, step, train_time)

    # Evaluate model on entire evaluation dataset, update metrics, save to log file
    validation_accuracy = metrics.test(model, eval_data_a, eval_data_b, step)

Usage after training (evaluating but not logging):
    metrics = Metrics(log_dir, num_classes, num_domains, al_config,
        None, None, generalize, domain_b_data is not None)

    # Evaluate on datasets
    metrics.train(model, train_data_a, train_data_b, evaluation=True)
    metrics.test(model, eval_data_a, eval_data_b, evaluation=True)

    # Get the results
    results = metrics.results()
"""
import time
import tensorflow as tf

from absl import flags

from load_data import tf_domain_labels

FLAGS = flags.FLAGS

class Metrics:
    """
    Handles keeping track of metrics either over one batch or many batch, then
    after all (or just the one) batches are processed, saving this to a log file
    for viewing in TensorBoard.

    Accuracy values:
        accuracy_{domain,task}/{source,target}/{training,validation}
        {auc,precision,recall}_task/{source,target}/{training,validation}
        accuracy_task_class_{class1name,...}/{source,target}/{training,validation}
        rates_class_{class1name,...}/{TP,FP,TN,FN}/{source,target}/{training,validation}
    Loss values:
        loss/{total,task,domain}
    """
    def __init__(self, log_dir, num_classes, num_domains, config,
            task_loss, domain_loss, generalize, target_domain=True):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.config = config
        self.datasets = ["training", "validation"]
        self.task_loss = task_loss if task_loss is not None else lambda y_true, y_pred, training: 0
        self.domain_loss = domain_loss if domain_loss is not None else lambda y_true, y_pred: 0
        self.generalize = generalize
        self.target_domain = target_domain # whether we have just source or both

        if not target_domain:
            self.domains = ["source"]
        else:
            self.domains = ["source", "target"]

        # Create all entire-batch metrics
        self.batch_metrics = {dataset: {} for dataset in self.datasets}
        for domain in self.domains:
            for dataset in self.datasets:
                for name in ["domain", "task"]:
                    n = "accuracy_%s/%s/%s"%(name, domain, dataset)
                    self.batch_metrics[dataset][n] = tf.keras.metrics.CategoricalAccuracy(name=n)

        for domain in self.domains:
            for dataset in self.datasets:
                n = "auc_task/%s/%s"%(domain, dataset)
                self.batch_metrics[dataset][n] = tf.keras.metrics.AUC(name=n)

                n = "precision_task/%s/%s"%(domain, dataset)
                self.batch_metrics[dataset][n] = tf.keras.metrics.Precision(name=n)

                n = "recall_task/%s/%s"%(domain, dataset)
                self.batch_metrics[dataset][n] = tf.keras.metrics.Recall(name=n)

        # Create all per-class metrics
        self.per_class_metrics = {dataset: {} for dataset in self.datasets}
        for i in range(self.num_classes):
            class_name = self.config.int_to_label(i)

            for domain in self.domains:
                for dataset in self.datasets:
                    n = "accuracy_task_class_%s/%s/%s"%(class_name, domain, dataset)
                    self.per_class_metrics[dataset][n] = tf.keras.metrics.Accuracy(name=n)

                    n = "rates_class_%s/TP/%s/%s"%(class_name, domain, dataset)
                    self.per_class_metrics[dataset][n] = tf.keras.metrics.TruePositives(name=n)

                    n = "rates_class_%s/FP/%s/%s"%(class_name, domain, dataset)
                    self.per_class_metrics[dataset][n] = tf.keras.metrics.FalsePositives(name=n)

                    n = "rates_class_%s/TN/%s/%s"%(class_name, domain, dataset)
                    self.per_class_metrics[dataset][n] = tf.keras.metrics.TrueNegatives(name=n)

                    n = "rates_class_%s/FN/%s/%s"%(class_name, domain, dataset)
                    self.per_class_metrics[dataset][n] = tf.keras.metrics.FalseNegatives(name=n)

        # Losses
        self.loss_total = tf.keras.metrics.Mean(name="loss/total")
        self.loss_task = tf.keras.metrics.Mean(name="loss/task")
        self.loss_domain = tf.keras.metrics.Mean(name="loss/domain")

    def _reset_states(self, dataset):
        """ Reset states of all the Keras metrics """
        for _, metric in self.batch_metrics[dataset].items():
            metric.reset_states()

        for _, metric in self.per_class_metrics[dataset].items():
            metric.reset_states()

        if dataset == "training":
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

    def _process_batch(self, results, domain, dataset):
        """ Update metrics for accuracy over entire batch for domain-dataset """
        task_y_true, task_y_pred, domain_y_true, domain_y_pred, \
            _, _, _ = results

        domain_names = [
            "accuracy_domain/%s/%s",
        ]

        for n in domain_names:
            name = n%(domain, dataset)
            self.batch_metrics[dataset][name](domain_y_true, domain_y_pred)

        task_names = [
            "accuracy_task/%s/%s",
            "auc_task/%s/%s",
            "precision_task/%s/%s",
            "recall_task/%s/%s",
        ]

        for n in task_names:
            name = n%(domain, dataset)
            self.batch_metrics[dataset][name](task_y_true, task_y_pred)

    def _process_per_class(self, results, domain, dataset):
        """ Update metrics for accuracy over per-class portions of batch for domain-dataset """
        task_y_true, task_y_pred, _, _, _, _, _ = results
        batch_size = tf.shape(task_y_true)[0]

        # If only predicting a single class (using softmax), then look for the
        # max value
        # e.g. [0.2 0.2 0.4 0.2] -> [0 0 1 0]
        per_class_predictions = tf.one_hot(
            tf.argmax(task_y_pred, axis=-1), self.num_classes)

        # List of per-class task metrics to update
        task_names = [
            "accuracy_task_class_%s/%s/%s",
            "rates_class_%s/TP/%s/%s",
            "rates_class_%s/FP/%s/%s",
            "rates_class_%s/TN/%s/%s",
            "rates_class_%s/FN/%s/%s",
        ]

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

            # Update metrics
            for n in task_names:
                name = n%(class_name, domain, dataset)
                self.per_class_metrics[dataset][name](acc_y_true, acc_y_pred)

    def _write_data(self, step, dataset, eval_time, train_time=None):
        """ Write either the training or validation data """
        assert dataset in self.datasets, "unknown dataset "+str(dataset)

        # Write all the values to the file
        with self.writer.as_default():
            for key, metric in self.batch_metrics[dataset].items():
                tf.summary.scalar(key, metric.result(), step=step)

            for key, metric in self.per_class_metrics[dataset].items():
                tf.summary.scalar(key, metric.result(), step=step)

            tf.summary.scalar("step_time/metrics/%s"%(dataset), eval_time, step=step)

            if train_time is not None:
                tf.summary.scalar("step_time/%s"%(dataset), train_time, step=step)

            # Only log losses on training data
            if dataset == "training":
                tf.summary.scalar("loss/total", self.loss_total.result(), step=step)
                tf.summary.scalar("loss/task", self.loss_task.result(), step=step)
                tf.summary.scalar("loss/domain", self.loss_domain.result(), step=step)

        # Make sure we sync to disk
        self.writer.flush()

    def _run_partial(self, model, data_a, data_b, dataset):
        """ Run all the data A/B through the model -- data_a and data_b
        should both be of type tf.data.Dataset """
        if data_a is not None:
            self._run_multi_batch(data_a, model, 0, "source", dataset)

        if self.target_domain and data_b is not None:
            self._run_multi_batch(data_b, model, 1, "target", dataset)

    def _run_batch(self, model, data_a, data_b, dataset):
        """ Run a single batch of A/B data through the model -- data_a and data_b
        should both be a tuple of (x, task_y_true, domain_y_true) """
        if data_a is not None:
            self._run_single_batch(*data_a, model, 0, "source", dataset)

        if self.target_domain and data_b is not None:
            self._run_single_batch(*data_b, model, 1, "target", dataset)

    def _run_multi_batch(self, data, *args, **kwargs):
        """ Evaluate model on all batches in the data (data is a tf.data.Dataset) """
        for x, task_y_true, domain_y_true in data:
            self._run_single_batch(x, task_y_true, domain_y_true, *args, **kwargs)

    @tf.function
    def _run_single_batch(self, x, task_y_true, domain_y_true, model, domain_num,
            domain_name, dataset_name):
        """
        Run a batch of data through the model. Call after_batch() afterwards:
            after_batch([labels_batch_a, task_y_pred, domains_batch_a, domain_y_pred,
                total_loss, task_loss, domain_loss], domain_name, dataset_name)

        Domain should be either 0 or 1 (if num_domains==2).
        """
        assert dataset_name in self.datasets, "unknown dataset "+str(dataset_name)
        assert domain_name in self.domains, "unknown domain "+str(domain_name)

        # Note: if you do x.shape[0] here, it'll give None on last batch and error
        batch_size = tf.shape(x)[0]

        # Match the number of examples we have since the domain_y_true
        # is meant for generalization, where it's a different number for
        # each home
        if not self.generalize:
            domain_y_true = tf_domain_labels(domain_num, batch_size, self.num_domains)

        # Evaluate model on data
        task_y_pred, domain_y_pred = model(x, training=False)

        # Calculate losses
        task_l = self.task_loss(task_y_true, task_y_pred, training=False)
        domain_l = self.domain_loss(domain_y_true, domain_y_pred)
        total_l = task_l + domain_l

        # Process this batch
        results = [
            task_y_true, task_y_pred, domain_y_true, domain_y_pred,
            total_l, task_l, domain_l,
        ]

        self._process_batch(results, domain_name, dataset_name)
        self._process_per_class(results, domain_name, dataset_name)

        # Only log losses on training data
        if dataset_name == "training":
            self._process_losses(results)

    def train(self, model, data_a, data_b, step=None, train_time=None, evaluation=False):
        """
        Call this once after evaluating on the training data for domain A and
        domain B

        Note: leave off step and train_time if evaluation=True and make sure
        data_a and data_b are the entire training datasets rathe than a single
        batch as when evaluation=False.
        """
        dataset = "training"
        self._reset_states(dataset)

        if not self.target_domain:
            data_b = None

        t = time.time()

        # evaluation=True is a tf.data.Dataset, otherwise a single batch
        if evaluation:
            self._run_partial(model, data_a, data_b, dataset)
        else:
            self._run_batch(model, data_a, data_b, dataset)

        t = time.time() - t

        if not evaluation:
            assert step is not None and train_time is not None, \
                "Must pass step and train_time to train() if evaluation=False"
            step = int(step)
            self._write_data(step, dataset, t, train_time)

    def test(self, model, eval_data_a, eval_data_b, step=None, evaluation=False):
        """
        Evaluate the model on domain A/B but batched to make sure we don't run
        out of memory

        Note: leave off step if evaluation=True

        Returns: source task validation accuracy
        """
        dataset = "validation"
        self._reset_states(dataset)

        if not self.target_domain:
            eval_data_b = None

        t = time.time()
        self._run_partial(model, eval_data_a, eval_data_b, dataset)
        t = time.time() - t

        # We use the validation accuracy to save the best model
        acc = self.batch_metrics["validation"]["accuracy_task/source/validation"]
        validation_accuracy = float(acc.result())

        if not evaluation:
            assert step is not None, "Must pass step to test() if evaluation=False"
            step = int(step)
            self._write_data(step, dataset, t)

        return validation_accuracy

    def results(self):
        """ Returns one dictionary of all the current metric results (floats) """
        results = {}

        for dataset in self.datasets:
            for key, metric in self.batch_metrics[dataset].items():
                results[key] = float(metric.result())

            for key, metric in self.per_class_metrics[dataset].items():
                results[key] = float(metric.result())

        results["loss/total"] = float(self.loss_total.result())
        results["loss/task"] = float(self.loss_task.result())
        results["loss/domain"] = float(self.loss_domain.result())

        return results
