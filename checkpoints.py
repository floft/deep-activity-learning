"""
Checkpoints
"""
import os
import tensorflow as tf

from absl import flags

from file_utils import get_best_valid_accuracy, write_best_valid_accuracy

FLAGS = flags.FLAGS

flags.DEFINE_integer("latest_checkpoints", 1, "Max number of latest checkpoints to keep")
flags.DEFINE_integer("best_checkpoints", 1, "Max number of best checkpoints to keep")

class CheckpointManager:
    """
    Keep both the latest and the best on validation data

    Latest stored in model_dir and best stored in model_dir/best
    Saves the best validation accuracy in log_dir/best_valid_accuracy.txt
    """
    def __init__(self, checkpoint, model_dir, log_dir):
        self.checkpoint = checkpoint
        self.log_dir = log_dir

        # Keep track of the latest for restoring interrupted training
        self.latest_manager = tf.train.CheckpointManager(
            checkpoint, directory=model_dir, max_to_keep=FLAGS.latest_checkpoints)

        # Keeps track of our best model for use after training
        best_model_dir = os.path.join(model_dir, "best")
        self.best_manager = tf.train.CheckpointManager(
            checkpoint, directory=best_model_dir, max_to_keep=FLAGS.best_checkpoints)

        # Restore best from file or if no file yet, set it to zero
        self.best_validation = get_best_valid_accuracy(self.log_dir)

        if self.best_validation is None:
            self.best_validation = 0.0

    def restore_latest(self):
        """ Restore the checkpoint from the latest one """
        self.checkpoint.restore(self.latest_manager.latest_checkpoint)

    def restore_best(self):
        """ Restore the checkpoint from the best one """
        self.checkpoint.restore(self.best_manager.latest_checkpoint)

    def save(self, step, validation_accuracy=None):
        """ Save the latest model. If validation_accuracy specified and higher
        than the previous best, also save this model as the new best one. """
        # Always save the latest
        self.latest_manager.save(checkpoint_number=step)

        # Only save the "best" if it's better than the previous best
        if validation_accuracy is not None:
            if validation_accuracy > self.best_validation:
                self.best_manager.save(checkpoint_number=step)
                self.best_validation = validation_accuracy
                write_best_valid_accuracy(self.log_dir, self.best_validation)
