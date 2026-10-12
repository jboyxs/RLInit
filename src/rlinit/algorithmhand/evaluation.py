"""Shared seed policy for evaluation after training."""

FINAL_EVAL_SEED_OFFSET = 20_000


def final_eval_seed(training_seed):
    """Return the first episode seed for every trainer's final evaluation."""
    return training_seed + FINAL_EVAL_SEED_OFFSET
