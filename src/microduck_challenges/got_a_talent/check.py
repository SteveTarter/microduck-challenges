"""Does a MicroDuck Got a Talent policy perform its timeline on the Arena's stage?

    uv run --frozen --group workshop python -m microduck_challenges.got_a_talent.check policy.onnx [--timeline FILE]

Plays the stage the way the Arena scores an entry (arena.evaluate, with this
folder's timeline.json by default), then checks the performance against the
timeline, keyframe by keyframe:

- no fall, by the event's own rule;
- every walking move (twist): over the keyframe, the robot covers at least
  MIN_FOLLOW of each commanded forward, sideways and turn amount (command x
  keyframe length), in the commanded direction, measured in the frame the
  robot faced when the keyframe began;
- every keyframe held 2 s or more: each head joint within 0.25 rad of the
  default pose plus the keyframe's head command, over the hold's last 0.5 s.

Prints a line per keyframe; exits 1 when anything fails, including a
performance that did not play to the end.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from arena import stage
from arena.constants import CTRL_DT, DEFAULT_POSE, JOINT_NAMES
from arena.evaluate import evaluate
from arena.manifest import load_event
from arena.policy import PolicyError, load_policy
from arena.timeline import Timeline, TimelineError, load_timeline

EVENT = "got-a-talent"
TIMELINE = Path(__file__).with_name("timeline.json")
MIN_FOLLOW = 0.25
HEAD_HOLD_S = 2.0
HEAD_WINDOW_S = 0.5
HEAD_TOLERANCE_RAD = 0.25
HEAD_JOINTS = [JOINT_NAMES.index(n) for n in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")]


def _frames(start_s: float, end_s: float) -> slice:
    """The trace frames of the control steps that start in [start_s, end_s)."""
    return slice(round(start_s / CTRL_DT), round(end_s / CTRL_DT))


def _yaw(quat) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _moves(trace, command, start_s: float, end_s: float) -> tuple[str, bool, float]:
    """What the robot did against the keyframe's twist, and the smallest share of it followed."""
    wanted = np.asarray(command[0:3]) * (end_s - start_s)
    if not np.any(wanted):
        return "moves: none commanded", True, 1.0
    a = round(start_s / CTRL_DT)
    b = min(round(end_s / CTRL_DT), len(trace) - 1)
    if b <= a:
        return "moves: the performance ended before this keyframe", False, 0.0
    yaw0 = _yaw(trace[a, 3:7])
    dx, dy = trace[b, 0:2] - trace[a, 0:2]
    forward = np.cos(yaw0) * dx + np.sin(yaw0) * dy
    left = -np.sin(yaw0) * dx + np.cos(yaw0) * dy
    turn = (_yaw(trace[b, 3:7]) - yaw0 + np.pi) % (2 * np.pi) - np.pi
    done = np.array([forward, left, turn])
    parts, worst = [], np.inf
    for name, unit, w, d in zip(("forward", "left", "turn"), ("m", "m", "rad"), wanted, done):
        if w == 0:
            continue
        share = d / w
        worst = min(worst, share)
        parts.append(f"{name} {d:+.2f}/{w:+.2f} {unit}")
    return "moves: " + ", ".join(parts), worst >= MIN_FOLLOW, float(worst)


def _head(trace, command, start_s: float, end_s: float) -> tuple[str, bool, float]:
    if end_s - start_s < HEAD_HOLD_S:
        return f"head: held under {HEAD_HOLD_S:g} s, not checked", True, 0.0
    angles = trace[_frames(end_s - HEAD_WINDOW_S, end_s)][:, [7 + j for j in HEAD_JOINTS]]
    if not len(angles):
        return "head: the performance ended before this hold", False, float("inf")
    target = DEFAULT_POSE[HEAD_JOINTS] + np.asarray(command[3:7])
    worst = float(np.abs(angles.mean(axis=0) - target).max())
    return f"head: off by {worst:.2f} rad", worst <= HEAD_TOLERANCE_RAD, worst


def report(trace, timeline: Timeline, termination) -> tuple[list[str], bool, float]:
    """A line per keyframe then the fall line, whether everything passed, and the
    smallest share of a commanded move the robot followed."""
    ends = list(timeline.times[1:]) + [timeline.duration_s]
    lines, passed, followed = [], True, 1.0
    for t, command, end in zip(timeline.times, timeline.commands, ends):
        moves, ok_moves, share = _moves(trace, command, t, end)
        head, ok_head, _ = _head(trace, command, t, end)
        passed &= ok_moves and ok_head
        followed = min(followed, share)
        lines.append(f"t={t:5.2f}  {moves}; {head}")
    fell = stage.fell_at(trace, termination)
    lines.append("no fall" if fell is None else f"fell at {fell:.2f} s")
    return lines, passed and fell is None, followed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a MicroDuck Got a Talent policy's performance on the Arena's stage.")
    parser.add_argument("onnx")
    parser.add_argument("--timeline", default=str(TIMELINE))
    args = parser.parse_args(argv)
    event = load_event(EVENT)
    try:
        timeline = load_timeline(Path(args.timeline), event.time_limit_s)
        result = evaluate(event, load_policy(Path(args.onnx)), timeline)
    except (TimelineError, PolicyError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    lines, passed, followed = report(result.trace, timeline, event.termination)
    passed = passed and result.seeds_finished == 1
    print("\n".join(lines))
    print(f"{'PASS' if passed else 'FAIL'}: smallest share of a move followed {followed:.0%}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
