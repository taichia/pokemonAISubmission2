import os

import torch

from nn_agent import load_model, mcts_agent, policy_step

USE_SEARCH = False

"""
Dragapult ex Deck
Advanced Level
This deck focuses on setting up multiple knockouts to take at least three Prize cards in a single turn with its Phantom Dive attack.
"""

# Load deck.csv in the dataset.
file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = [int(csv[i]) for i in range(60)]

# Load the trained neural-network weights directly from the training output (out/best.pth).
# Missing weights are a hard error -- this agent has no rule-based fallback.
weights_path = "out/best.pth"
if not os.path.exists(weights_path):
    weights_path = "/kaggle_simulations/agent/" + weights_path
if not os.path.exists(weights_path):
    raise FileNotFoundError(
        f"NN weights not found at '{weights_path}'. Train with ppo_training.py to produce "
        f"out/best.pth.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_model = load_model(weights_path, device)


def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    if USE_SEARCH:
        return mcts_agent(obs_dict, my_deck, nn_model)
    return policy_step(obs_dict, my_deck, nn_model)
