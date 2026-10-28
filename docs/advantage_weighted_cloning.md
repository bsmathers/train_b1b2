# Monte Carlo advantage-weighted cloning

This experiment continues from the completed selected-actor model without
changing that run. The source checkpoint is loaded twice: once as the immutable
policy/value anchor and once as the live model. Epoch475 is used only to recover
the architecture and observation metadata required by the existing loader; its
weights and outputs do not enter the objective.

For a logged state and action, let the terminal result from that player's
perspective be `G` in `{-1, 0, 1}`. The frozen source value head gives

    V(s) = P(win | s) - P(loss | s)

and the Monte Carlo advantage estimate is

    A(s, a_logged) = G - V(s).

The logged action receives weight

    w = clip(exp(A / 0.5), 0.25, 4.0).

The weighted negative log likelihood is divided by the mean weight in the
batch. This keeps the overall AdamW step scale comparable to ordinary behavior
cloning while changing the relative importance of decisions. The complete
actor objective is

    0.55 * normalized_AWR_NLL
      + 0.25 * KL(source_actor || live_actor)
      + 0.20 * KL(source_value || live_value).

The W/L/D head is frozen, while the actor and backbone remain trainable. The
training set is the existing selected strong-model subset. Padding and missing
actions never contribute to weights or loss.

The default run uses 1,000 AdamW optimizer steps per epoch, saves every epoch,
and stops after 15 consecutive validation epochs with no decrease. `MIN_DELTA`
defaults to zero so a slow but real decrease resets patience. Validation uses a
fixed held-out split to prevent sampling noise from choosing checkpoints.

Monitor `B1/AWR ESS Fraction` alongside policy KL, value KL,
`B1/Actor Objective Loss`, and the unweighted `B1/Actor BC Loss`. A persistently
very small ESS fraction means the update is
being driven by too few decisions; increase `awr_temperature` or tighten the
weight bounds before trusting such a run. Final gameplay selection still
requires a fresh, paired tournament against the unmodified source model.
