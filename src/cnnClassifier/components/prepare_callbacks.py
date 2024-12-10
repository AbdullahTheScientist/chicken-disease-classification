import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from src.cnnClassifier.entity.config_entity import PrepareCallbacksConfig



class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    
    @property
    def _create_ckpt_callbacks(self):
        # Ensure filepath ends with '.h5' or '.keras'
        checkpoint_filepath = str(self.config.checkpoint_model_filepath)
        if not (checkpoint_filepath.endswith(".h5") or checkpoint_filepath.endswith(".keras")):
            raise ValueError("The filepath must end with '.h5' or '.keras'.")

        return tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True  # Save only the best model
        )

    def get_tb_ckpt_callbacks(self):
        """
        Returns a list of TensorBoard and ModelCheckpoint callbacks.
        """
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]