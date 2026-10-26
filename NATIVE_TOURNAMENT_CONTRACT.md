# Native Gen 1 tournament: simulation, policy inputs, and action contract

This describes the **native Zig tournament** (`tools/select_stage1_b1_epoch.py --transport native`), not
the offline B1 gradient-update loop. Native battle simulation uses the Zig
`@pkmn/engine` addon in `metamon/env/vectorized/battle_host_engine.js`; there is no
WebSocket Showdown server.

## Exact per-decision pipeline

1. `VectorizedShowdownEnv` runs N independent battles. Each lane has its own
   `StreamBattleLane` with *two* `MetamonBackendBattle` parsers (`p1` and
   `p2`) and a separate `GroupedObservationSpace` instance for each side.
2. Zig's turn result and protocol events are decoded by `battle_host_engine.js`. Public events are dispatched to both parsers;
   `|request|` messages contain only the request for the relevant side.
   A request's `side.id` and Pokémon identification prefixes must match
   its recipient side.
3. Each side independently constructs `UniversalState.from_Battle(...)` and
   `GroupedObservationSpace.state_to_obs(...)`. This contains current HP,
   statuses, types, known moves and PP, available switches, the opponent's
   visible active Pokémon, field conditions, previous moves, and tracked
   revealed-opponent species. It is **not** the raw full battle log, nor an
   omniscient game state. There is a 13-dimensional `illegal_actions`
   mask computed from the same side's legal action set.
4. The model's `AmagoLadderPolicyDriver.act(active, obs_list)` receives
   `obs`, `rl2s`, `time_idxs` and the **per-lane transformer KV
   cache**. The actor is the configured `MetamonMaskedResidualActor` with
   `pass_obs_keys_to_actor = ["illegal_actions"]`. Evaluation runs with
   `sample=True` (categorical policy sampling), rather than deterministic
   greedy action selection.
5. `UniversalAction.action_idx_to_BattleOrder` decodes the model's
   discrete action in Metamon's **canonical sorted move/switch order**. It
   becomes a Showdown-style string via `metamon/env/vectorized/action_adapter.py` and then the native Zig
   `Choice` via `native_choice.js`. All explicit choices MUST match a
   native `battle.choices(side, result)` entry. Unexpected values are fatal
   rather than silently replaced with a different legal action.
6. After resolution, each model's `observe(lane, reward, committed_action)`
   advances its timestep and sets RL2 input for the **next** decision to
   `[reward, one_hot(previous_action)]`. The transformer KV history is
   retained for that lane, reset at episode boundaries, and not shared
   with the other model or other lanes. Opponent-only decisions must record
   the same RL2 feedback and advance the opponent cache.
7. The engine emits an authoritative win/lose/tie; the tournament records
   1 / 0 / 0.5 from B1's point of view. It scores fixed game quotas per lane,
   all four good/bad team combinations, and both physical p1/p2 positions.

## Actions and side effects

`DefaultActionSpace` has 13 outputs:

| Policy output | Meaning |
|---|---|
| 0–3 | Move slots in **Metamon's consistent sorted move order**, not necessarily the engine's in-party move order |
| 4–8 | Switches to one of five available party members in **Metamon's consistent sorted species order**, not their native party slots |
| 9–12 | Terastallization variants of the four moves; irrelevant in Gen 1 and not legitimate legal outputs |

The Python action adapter resolves the selected move to a Showdown move
name; switch names are resolved to **1-based party positions** from the
private request. The native resolver then looks up the corresponding engine
`Choice.move(slot)` or `Choice.switch(slot)` and verifies it is legal. **These
switch index systems are different:** Showdown `switch N` addresses the
original party position; Zig keeps the active Pokémon in dynamic slot 1,
so a native `Choice.switch` can only target **2–6**. The adapter translates
`side.slot(originalPosition)` to the dynamic slot before executing a switch.
Private request Pokémon are sorted by their stable `Pokemon.position`,
rather than Zig's reordered `side.pokemon` iteration order.

Crucially, switching **back into the initial lead** uses **Showdown**
`switch 1`, which is translated into the appropriate **native dynamic
slot 2–6**. Sending a native `Choice.switch(1)` is invalid. Showdown party
slots **1 through 6** are supported. A failure to translate
an explicit action is an **evaluation error**, not permission to auto-select
another move. The special `default` request is the only intentional
first-legal-action fallback (e.g., automatic team preview); forced
`recharge`, `struggle` and `fight` aliases can choose a sole legal move.

The evaluator sets:

- `METAMON_REQUIRE_PKMN_ENGINE=1`: never silently fall back to non-Zig simulation
- `METAMON_NATIVE_STRICT_ACTIONS=1`: fail if a model output is not in
  the observation's legal action set or cannot be converted to a move
- `METAMON_NATIVE_STRICT_PARSER=1`: fail if a protocol event cannot
  be parsed instead of discarding historical state changes

One can explicitly set a strict variable to `0` to debug legacy behavior,
but **such results must not be used for checkpoint selection**.

## Training versus evaluation

`tools/train_stage1_b1_cuda.sh` invokes `metamon.rl.finetune` and
`stage1_b1_strong.yaml`, with `replay_weight: 0.0` and curated recorded
trajectories. This is **offline imitation/critic training**, not rollout
collection through the native simulator. Adapter bugs in live matches do
**not** directly alter its recorded training trajectories or optimizer
updates. This does not prove the dataset's labels, model weights, or
training objective are error-free.

The actor is trained against recorded moves; reported validation
imitation NLL measures agreement with held-out records, **not** strength
against TaurosV0. Do not interpret a low live win rate alone as proof
that training malfunctioned.

## Validation matrix

Run these from the repository root:

```bash
node --test tests/test_native_choice.cjs
./tools/setup_pkmn_engine.sh
node tests/native_sim_integration.cjs
python3 -m unittest discover -s tests -p 'test_stage1_selection.py' -v
python3 -m unittest discover -s tests -p 'test_native_setup.py' -v
```

The first test checks all four move slots, all six native party slots
(including slot 1), species/move-name translation, illegal and ambiguous
requests, and explicit default/forced-action behavior. The native integration
test compiles/loads the *real* Zig addon, switches out of the original
party slot 1, then issues Showdown `switch 1` and verifies it maps to a
legal dynamic native slot. Stage-1 CI executes those tests.

**Remaining scope requiring separate validation on the training machine:**
policy-weighted matches on the actual CUDA device (B1 vs V0, V1@475
vs V0, and V0 self-play), observation parity against a standard Showdown
server on identical seeded games, and end-to-end trajectory comparison with
training replays. Local tests and CI alone cannot certify those.
