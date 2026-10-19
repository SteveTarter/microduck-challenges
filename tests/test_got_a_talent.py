"""MicroDuck Got a Talent trains in the Arena's physics, and reads the command a timeline carries."""

import importlib
import math
from pathlib import Path

import numpy as np
import pytest

import microduck_challenges  # noqa: F401  (registers every task)
from arena.assets import ASSET_ROOT
from arena.constants import CTRL_DT, DECIMATION, DEFAULT_POSE, JOINT_NAMES, TRACE_WIDTH
from arena.course import compile_course
from arena.manifest import load_event
from arena.timeline import Timeline, load_timeline
from mjlab_microduck.tasks import mdp as microduck_mdp


@pytest.fixture(scope="module")
def cfg():
    return importlib.import_module("microduck_challenges.got_a_talent.env").make_env_cfg()


@pytest.fixture(scope="module")
def arena():
    return compile_course(load_event("got-a-talent"))


def _training_model(entity):
    """The robot as training compiles it: its MJCF with the entity's collision
    settings applied (mjlab's Entity runs each CollisionCfg.edit_spec on the spec)."""
    spec = entity.spec_fn()
    for collision in entity.collisions:
        collision.edit_spec(spec)
    return spec.compile()


def _body(model, geom):
    return model.body(int(model.geom_bodyid[geom])).name


def _robot_geoms(model, skip):
    return [g for g in range(model.ngeom)
            if (model.geom_contype[g] or model.geom_conaffinity[g]) and _body(model, g) not in skip]


def _colliding(model, skip):
    """(body, geom type) of every geom that can collide, bodies in `skip` left out."""
    return sorted((_body(model, g), int(model.geom_type[g])) for g in _robot_geoms(model, skip))


# mjlab's flat terrain (terrain_entity._import_ground_plane) is a plane with
# MuJoCo's default contact: priority 0, condim 3, sliding friction 1.
FLOOR = (0, 3, 1.0)


def _floor_contacts(model, skip):
    """(body, condim, sliding friction) of each robot geom's contact with that floor.
    MuJoCo pairs two geoms by priority first; at equal priority it takes the larger
    condim and the larger friction."""
    out = []
    for g in _robot_geoms(model, skip):
        priority, condim, friction = int(model.geom_priority[g]), int(model.geom_condim[g]), float(model.geom_friction[g][0])
        if priority > FLOOR[0]:
            pair = (condim, friction)
        elif priority < FLOOR[0]:
            pair = FLOOR[1:]
        else:
            pair = (max(condim, FLOOR[1]), max(friction, FLOOR[2]))
        out.append((_body(model, g), *pair))
    return sorted(out)


def test_got_a_talent_collides_like_the_arena(cfg, arena):
    # A sit puts the trunk and hips on the floor: training must collide with the
    # same robot geoms as the Arena's stage, or a seated trunk sinks in one and
    # not the other.
    robot = _training_model(cfg.scene.entities["robot"])
    assert _colliding(robot, skip={"world"}) == _colliding(arena, skip={"world", "course"})


def test_got_a_talent_touches_the_floor_like_the_arena(cfg, arena):
    """Every robot geom meets the floor with the Arena's condim and friction.
    Robot-on-robot contacts are where the two differ: training gives its non-foot
    geoms condim 1 (FULL_COLLISION), so two of them touch without friction, and
    the Arena's are all condim 3."""
    floor = arena.geom("floor").id
    assert (int(arena.geom_priority[floor]), int(arena.geom_condim[floor]), float(arena.geom_friction[floor][0])) == FLOOR
    robot = _training_model(cfg.scene.entities["robot"])
    assert _floor_contacts(robot, skip={"world"}) == _floor_contacts(arena, skip={"world", "course"})


def test_got_a_talent_solves_contacts_like_the_arena(cfg, arena):
    assert cfg.sim.mujoco.iterations == arena.opt.iterations
    assert cfg.sim.mujoco.ls_iterations == arena.opt.ls_iterations
    assert math.isclose(cfg.sim.mujoco.timestep, arena.opt.timestep)
    assert cfg.decimation == DECIMATION


def test_got_a_talent_ends_a_training_episode_at_a_fall(cfg):
    """dance1: a walker cannot get up, so training ends an episode at a fall, as
    every walking task does. The stage still plays on after one (the Arena's rule)."""
    assert "fell_over" in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_got_a_talent_cannot_rest_lying_down(cfg):
    """dance2: the Arena's collision set lets the trunk rest on the floor. A fall
    must end the episode where the Arena calls one (60°, gravity_z -0.5), and a
    trunk on the floor must end it whatever its tilt, or lying still pays."""
    limit = cfg.terminations["fell_over"].params["limit_angle"]
    event = load_event("got-a-talent")
    assert math.isclose(math.cos(limit), -event.termination.fall_gravity_z)
    assert cfg.terminations["trunk_down"].params["minimum_height"] >= 0.06


def test_got_a_talent_reads_what_a_timeline_carries(cfg):
    """twist the walking command (forward, sideways, turn), head_pose the 4 head
    deltas, body zeros: a keyframe's 13 numbers, in the 61-D observation's command slot."""
    assert isinstance(cfg.commands["twist"], microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert len(cfg.commands["head_pose"].ranges) == 4
    actor = cfg.observations["actor"].terms
    assert list(actor)[-3:] == ["command", "head_command", "body_command"]
    assert actor["head_command"].params == {"command_name": "head_pose"}


def test_checks_head_joint_order_matches_head_pose_trackings_own(cfg):
    """check.HEAD_JOINTS reads the head command's 4 dims as
    [neck_pitch, head_pitch, head_yaw, head_roll]; that must be the same order
    head_pose_tracking's own find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
    resolves them in, given this robot's actual joint declaration order."""
    from microduck_challenges.got_a_talent.check import HEAD_JOINTS

    entity = cfg.scene.entities["robot"].build()
    _, names = entity.find_joints_by_actuator_names(microduck_mdp._NECK_JOINT_PATTERNS)
    assert [JOINT_NAMES[i] for i in HEAD_JOINTS] == names


TIMELINE = Path(microduck_challenges.__file__).parent / "got_a_talent" / "timeline.json"


def test_the_reference_timeline_plays_on_the_arena():
    timeline = load_timeline(TIMELINE, load_event("got-a-talent").time_limit_s)
    assert timeline.duration_s == 20.0


def test_the_reference_timeline_stays_inside_what_training_saw(cfg):
    """Every walking command within the trained ranges, the head within its
    starting range (the curriculum widens it later), body zeros."""
    timeline = load_timeline(TIMELINE, load_event("got-a-talent").time_limit_s)
    ranges = cfg.commands["twist"].ranges
    head = cfg.commands["head_pose"].ranges
    for command in timeline.commands:
        for value, (lo, hi) in zip(command[0:3], (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)):
            assert lo <= value <= hi
        for value, (lo, hi) in zip(command[3:7], head):
            assert lo <= value <= hi
        assert command[7:] == (0.0,) * 6


STILL = (0.0,) * 13
LEFT = (0.0, 0.3, 0.0) + (0.0,) * 10
TURN = (0.0, 0.0, 0.5) + (0.0, 0.0, 0.5, 0.0) + (0.0,) * 6
EIGHT_S = Timeline(8.0, (0.0, 2.0, 4.0), (STILL, LEFT, TURN))


def _trace(y_at=lambda t: 0.0, yaw_at=lambda t: 0.0, head_yaw_at=lambda t: 0.0, tilt_at=lambda t: 0.0):
    """A synthetic performance: sideways position, heading, head yaw delta and roll tilt over time."""
    n = round(8.0 / CTRL_DT)
    trace = np.zeros((n, TRACE_WIDTH), dtype=np.float32)
    for i in range(n):
        t = i * CTRL_DT
        yaw, roll = yaw_at(t) / 2, np.radians(tilt_at(t)) / 2
        # roll about x, then yaw about z (w, x, y, z)
        trace[i, 3:7] = (np.cos(yaw) * np.cos(roll), np.cos(yaw) * np.sin(roll),
                         np.sin(yaw) * np.sin(roll), np.sin(yaw) * np.cos(roll))
        trace[i, 1] = y_at(t)
        trace[i, 2] = 0.115
        trace[i, 7:] = DEFAULT_POSE
        trace[i, 7 + JOINT_NAMES.index("head_yaw")] += head_yaw_at(t)
    return trace


def _slide_left(t):
    return 0.3 * min(max(t - 2.0, 0.0), 2.0)


def _turn_left(t):
    return 0.5 * min(max(t - 4.0, 0.0), 4.0)


def _report(trace):
    from microduck_challenges.got_a_talent import check

    return check.report(trace, EIGHT_S, load_event("got-a-talent").termination)


def test_a_performance_that_follows_its_timeline_passes():
    lines, passed, followed = _report(_trace(_slide_left, _turn_left, lambda t: 0.5 if t >= 4.5 else 0.0))
    assert passed, lines
    assert len(lines) == 4  # three keyframes, then the fall line
    assert followed > 0.9


def test_standing_still_through_a_move_fails():
    lines, passed, followed = _report(_trace(yaw_at=_turn_left, head_yaw_at=lambda t: 0.5 if t >= 4.5 else 0.0))
    assert not passed
    assert followed == pytest.approx(0.0, abs=1e-3)
    assert "left +0.00/+0.60 m" in lines[1]


def test_moving_the_wrong_way_fails():
    lines, passed, followed = _report(_trace(lambda t: -_slide_left(t), _turn_left, lambda t: 0.5 if t >= 4.5 else 0.0))
    assert not passed
    assert followed < 0


def test_a_head_far_from_its_command_fails():
    lines, passed, _ = _report(_trace(_slide_left, _turn_left, lambda t: 0.9 if t >= 4.5 else 0.0))
    assert not passed
    assert "head: off by 0.40 rad" in lines[2]


def test_a_fall_fails():
    lines, passed, _ = _report(_trace(_slide_left, _turn_left, lambda t: 0.5 if t >= 4.5 else 0.0,
                                      tilt_at=lambda t: 80.0 if t >= 6.0 else 0.0))
    assert not passed
    assert lines[-1] == "fell at 6.30 s"


def test_check_plays_the_stage_end_to_end(capsys):
    """End to end on the Arena, with Pollen's walker bundled in the Arena."""
    from microduck_challenges.got_a_talent import check

    check.main([str(ASSET_ROOT / "policies" / "BEST_alpha_walking.onnx")])
    out = capsys.readouterr().out.splitlines()
    timeline = load_timeline(TIMELINE, load_event("got-a-talent").time_limit_s)
    assert len(out) == len(timeline.times) + 2  # a line per keyframe, the fall line, the verdict
    assert out[-1].split(":")[0] in ("PASS", "FAIL")


def test_check_refuses_a_malformed_timeline(tmp_path, capsys):
    """main() must fail cleanly, like the Arena's own CLI, instead of raising."""
    from microduck_challenges.got_a_talent import check

    bad = tmp_path / "timeline.json"
    bad.write_text("{nope")
    assert check.main(["policy.onnx", "--timeline", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "timeline.json" in err and "not a JSON file" in err


def test_the_manifest_says_what_the_policy_reads():
    """publish writes this into every got-a-talent manifest: the walking command and the head pose, never 'unused'."""
    from mjlab_microduck import challenge as ch

    import microduck_challenges  # noqa: F401 — registers every challenge's task

    command = ch.for_task("Mjlab-GotATalent-MicroDuck").command
    assert command["twist"].startswith("[forward m/s")
    assert "head_yaw" in command["head"]
