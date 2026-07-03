# Register dataset classes here
from research.datasets.replay_buffer import ReplayBuffer
from research.datasets.feedback_buffer import (
    PairwiseComparisonDataset,
    ReplayAndFeedbackBuffer,
    MW_FeedbackDataset,
    D4RL_FeedbackDataset,
)
from research.datasets.d4rl_dataset import D4RL_OfflineDataset
from research.datasets.mw_dataset import MW_OfflineDataset

