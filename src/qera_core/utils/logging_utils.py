# src/qera_core/utils/logging_utils.py
import tensorflow as tf
import datetime
import os

# For Weights & Biases (optional, but highly recommended for complex RL)
# import wandb
# wandb.init(project="qera-phase1-rl", entity="your_wandb_entity") # Configure in main script

def setup_tensorboard_logger(log_dir: str): # <--- THIS IS THE FUNCTION NAME IT'S LOOKING FOR
    """Sets up a TensorBoard logger."""
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, current_time)
    summary_writer = tf.summary.create_file_writer(log_dir)
    return summary_writer

def log_episode_metrics(summary_writer, episode: int, total_reward: float, fidelity_at_end: float, **kwargs): # <--- THIS IS THE FUNCTION NAME IT'S LOOKING FOR
    """Logs metrics to TensorBoard."""
    with summary_writer.as_default():
        tf.summary.scalar('episode_reward', total_reward, step=episode)
        tf.summary.scalar('final_fidelity', fidelity_at_end, step=episode)
        for key, value in kwargs.items():
            tf.summary.scalar(key, value, step=episode)
    summary_writer.flush()
    # If using wandb:
    # wandb.log({"episode_reward": total_reward, "final_fidelity": fidelity_at_end, **kwargs}, step=episode)
