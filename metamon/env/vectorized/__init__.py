"""Vectorized native simulation env for Metamon.

Runs Gen 1 battles through the native @pkmn/engine Zig simulator and batches the in-the-loop opponent's
neural-network inference across lanes. By default the evaluated agent plays Showdown
``p1``; pass ``eval_player_side=1`` for ``p2``. Unsupported formats retain the
bundled Pokemon Showdown fallback.
"""

from .lane import StreamBattleLane
from .amago_policy import AmagoLadderPolicyDriver, vectorized_ladder_eval
from .opponent import (
    AmagoBatchedOpponent,
    BatchedOpponent,
    ConfigBatchedOpponent,
    RandomBatchedOpponent,
)
from .sim_process import ShowdownSimProcess, ShowdownSimProcessError, make_sim_process
from .vector_env import (
    BattleAgainstMetamon,
    BattleAgainstOpponentPool,
    BattleShowdownVectorized,
    ShowdownEnv,
    VectorizedShowdownEnv,
)

__all__ = [
    "StreamBattleLane",
    "ShowdownSimProcess",
    "ShowdownSimProcessError",
    "make_sim_process",
    "BatchedOpponent",
    "RandomBatchedOpponent",
    "AmagoBatchedOpponent",
    "AmagoLadderPolicyDriver",
    "ConfigBatchedOpponent",
    "vectorized_ladder_eval",
    "VectorizedShowdownEnv",
    "ShowdownEnv",
    "BattleAgainstMetamon",
    "BattleAgainstOpponentPool",
    "BattleShowdownVectorized",
]
