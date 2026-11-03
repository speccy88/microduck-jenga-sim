"""Vectorized, backend-neutral fast-strike criterion for this simulation only.

Use torch in the environment and NumPy for CPU regression tests of the same
function. Thresholds define experimental behavior, not safe hardware limits.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StrikeCriteria:
    min_head_forward_m_s: float = 0.25
    max_total_contact_s: float = 0.12
    min_top_separation_m: float = 0.03
    max_separation_delay_s: float = 0.30
    max_precontact_top_motion_m: float = 0.01


CRITERIA = StrikeCriteria()


def criteria_dict():
    return asdict(CRITERIA)


def initial_strike_state(zero, vector_zero):
    state = {name: zero * 1.0 for name in (
        "strike_elapsed_s", "strike_samples", "strike_seen_contact",
        "strike_previous_forward_m_s", "strike_head_speed_m_s",
        "strike_peak_head_forward_m_s", "strike_contact_duration_s",
        "strike_precontact_displaced", "strike_invalid", "strike_candidate",
        "strike_separated", "fast_strike")}
    state.update(strike_first_contact_s=zero - 1.0,
                 strike_separation_delay_s=zero - 1.0,
                 strike_previous_mouth_b=vector_zero * 1.0,
                 strike_previous_top_delta=vector_zero * 1.0)
    return state


def update_strike(state, mouth_b, toward_top_b, top_delta, contact, dt, xp,
                  criteria=CRITERIA):
    """Advance once per distinct 50 Hz environment step.

    mouth_b is the mouth-tip position in the trunk frame. Its finite difference
    excludes rigid trunk translation/rotation: walking into the block with a
    stationary neck cannot satisfy this criterion. Speed is the larger of the
    first-contact sample and its immediately preceding sample, to include the
    pre-impact approach. Contacts shorter than a 20 ms sample can be missed.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    s = dict(state)
    now = state["strike_elapsed_s"] + dt
    sampled = state["strike_samples"] > 0
    finite = (xp.isfinite(mouth_b).all(axis=-1)
              & xp.isfinite(toward_top_b).all(axis=-1)
              & xp.isfinite(top_delta).all(axis=-1))
    velocity = (mouth_b - state["strike_previous_mouth_b"]) / dt
    forward = xp.sum(velocity * toward_top_b, axis=-1)
    forward = xp.where(finite & sampled, xp.maximum(forward, forward * 0), 0.0)
    first = contact & (state["strike_seen_contact"] == 0)
    first_speed = xp.maximum(forward, state["strike_previous_forward_m_s"])
    old_top = state["strike_previous_top_delta"]
    prior_xy = xp.sqrt(xp.sum(old_top[:, :2] ** 2, axis=-1))
    prior_motion = (prior_xy > criteria.max_precontact_top_motion_m) | (
        xp.abs(old_top[:, 2]) > criteria.max_precontact_top_motion_m)
    s["strike_invalid"] = xp.maximum(state["strike_invalid"], xp.where(~finite, 1.0, 0.0))
    s["strike_precontact_displaced"] = xp.where(first, xp.where(prior_motion | ~sampled, 1.0, 0.0),
                                                state["strike_precontact_displaced"])
    s["strike_first_contact_s"] = xp.where(first, now, state["strike_first_contact_s"])
    s["strike_head_speed_m_s"] = xp.where(first, first_speed, state["strike_head_speed_m_s"])
    s["strike_seen_contact"] = xp.maximum(state["strike_seen_contact"], xp.where(contact, 1.0, 0.0))
    s["strike_contact_duration_s"] = state["strike_contact_duration_s"] + xp.where(contact, dt, 0.0)
    s["strike_candidate"] = xp.where(
        (s["strike_seen_contact"] > 0)
        & (s["strike_head_speed_m_s"] >= criteria.min_head_forward_m_s)
        & (s["strike_precontact_displaced"] == 0) & (s["strike_invalid"] == 0), 1.0, 0.0)
    xy = xp.sqrt(xp.sum(top_delta[:, :2] ** 2, axis=-1))
    delay = now - s["strike_first_contact_s"]
    separates = ((s["strike_seen_contact"] > 0)
                 & (xy >= criteria.min_top_separation_m)
                 & (delay <= criteria.max_separation_delay_s + 1e-9) & finite)
    newly_separated = separates & (state["strike_separated"] == 0)
    s["strike_separated"] = xp.maximum(state["strike_separated"], xp.where(separates, 1.0, 0.0))
    s["strike_separation_delay_s"] = xp.where(newly_separated, delay, state["strike_separation_delay_s"])
    s["fast_strike"] = xp.where(
        (s["strike_candidate"] > 0) & (s["strike_separated"] > 0)
        & (s["strike_contact_duration_s"] <= criteria.max_total_contact_s + 1e-9), 1.0, 0.0)
    s["strike_peak_head_forward_m_s"] = xp.maximum(state["strike_peak_head_forward_m_s"], forward)
    s["strike_elapsed_s"], s["strike_samples"] = now, state["strike_samples"] + 1
    s["strike_previous_forward_m_s"] = forward
    s["strike_previous_mouth_b"] = mouth_b * 1.0
    s["strike_previous_top_delta"] = top_delta * 1.0
    return s
