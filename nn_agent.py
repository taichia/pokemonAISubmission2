"""Submission-side agent: load weights and pick an action from an observation.

main.py imports load_model, policy_step (and mcts_agent only if USE_SEARCH).
At inference the policy is greedy over legal candidate actions -- no forward search,
so no hidden opponent state ever needs to be guessed.
"""

import torch

from cg.api import to_observation_class
from features import (
    MyModel,
    enumerate_actions,
    evaluate,
    get_decoder_input,
    get_encoder_input,
)

# Architecture must match what ppo_training.py saved. Kept in the checkpoint so we
# can rebuild the right shape regardless of hyperparameter changes.
_DEFAULT_ARCH = dict(d_model=128, num_heads=2, d_feedforward=256,
                     num_layers_encoder=2, num_layers_decoder=2)


def load_model(weights_path: str, device: torch.device) -> MyModel:
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        arch = ckpt.get("arch", _DEFAULT_ARCH)
        state = ckpt["state_dict"]
    else:
        arch = _DEFAULT_ARCH
        state = ckpt
    model = MyModel(**arch).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def policy_step(obs_dict: dict, your_deck: list[int], model: MyModel) -> list[int]:
    """Greedy action selection: highest-logit legal candidate."""
    obs = to_observation_class(obs_dict)
    device = next(model.parameters()).device
    actions = enumerate_actions(obs)
    sv_enc = get_encoder_input(obs, your_deck)
    sv_dec = get_decoder_input(obs, actions)
    _value, logits = evaluate(model, sv_enc, sv_dec, device)
    best = int(torch.argmax(logits[: len(actions)]).item())
    return actions[best]


# Optional: only used when main.py sets USE_SEARCH = True. The model-free policy is
# the intended path; this is a thin stub so the import in main.py never fails.
def mcts_agent(obs_dict: dict, your_deck: list[int], model: MyModel) -> list[int]:
    return policy_step(obs_dict, your_deck, model)
