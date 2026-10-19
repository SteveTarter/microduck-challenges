# AGENTS.md: MicroDuck Got a Talent

You are helping a user prepare a Microduck performance for the Arena's **MicroDuck Got a Talent**
(event `got-a-talent`, task `Mjlab-GotATalent-MicroDuck`, this folder). The user may be new to
reinforcement learning. Work so that it stays:

- **simple**: the user decides the performance, the budget and what gets published; you do the
  mechanics;
- **transparent**: say what you change and why before each run or edit, one commit per run, and
  report what the Arena's stage and `check.py` measure, never an impression from the curves;
- **efficient**: a better performance often needs no training at all (edit `timeline.json`); when
  it does, one change per run, a smoke test before any long run, every checkpoint judged.

Never launch a Hugging Face job, publish, enter the Arena or push without the user's explicit yes.

## The challenge

A performance, not a race. The rules, from `arena/events/got-a-talent.toml` in the installed Arena:

- **Stage**: a flat, open floor, no walls. Start from the `STAND` pose, 0.5 s settle (playing the
  first keyframe), joints perturbed by ±0.01 rad, heading by ±2°.
- **Performance**: the Arena plays the entry's `timeline.json` once, for at most **20 s**. Nothing
  is timed and nothing has to be finished. A fall does not stop it: it is part of the show (the fall
  rule, trunk tilted more than 60° for 0.3 s, only labels it). An entry without a timeline stands
  still for 20 s.
- **Judging**: visitors vote. Every performance played to the end goes to the ballots: two to four
  anonymous performances, watched one at a time on stage. The most votes over the **last seven
  days** wins; ties favour the earlier publication. The user's **latest** performance played to the
  end represents them, and votes on an older entry do not move to its replacement.

So what wins is a performance people want to watch: clear moves, timing, a little story, executed
reliably by the policy.

## The policy and the timeline

61 observations → 14 joint position targets at 50 Hz, exported to ONNX (≤ 64 MiB); `publish` checks
the shape. The observations, as the Arena builds them: gyro (3), projected gravity (3), joint
positions from the default pose (14), joint velocities one control step old (14), last action (14),
then the 13 command numbers the timeline gives: twist (3), head (4), body (6). This challenge's
policy reads:

- `twist`, **the walking command**: forward m/s, sideways m/s (left +), turn rad/s (left +);
- `head`, **4 angles in radians from the default pose**: neck_pitch, head_pitch, head_yaw, head_roll;
- `body`: unused (zeros).

Since dance1 (2026-10-19) the policy is a walker, and this folder's `timeline.json` is the
**Electric Slide** (Ric Silvern, 18 counts at 0.5 s each, so a wall is 9 s): vine right, vine left,
move back, forward touch, back touch, then step and quarter turn left, repeated. Each section is one
keyframe:

```json
{"timeline_version": 1, "duration_s": 20.0, "keyframes": [
  {"t": 0, "twist": [0, -0.3, 0], "head": [0, 0, 0, 0]},
  {"t": 2, "twist": [0, 0.3, 0],  "head": [0, 0, 0, 0]},
  {"t": 4, "twist": [-0.3, 0, 0], "head": [0, 0, 0, 0]},
  {"t": 8, "twist": [0.15, 0, 1], "head": [0, 0, 0, 0]}
]}
```

From each keyframe's `t` until the next one, the policy reads that keyframe's numbers. A group a
keyframe leaves out is zeros, not the previous value. The first keyframe is at 0, times go up, at most
1000 keyframes, every number within ±10, `duration_s` at most 20.

**What the policy can do is what training showed it.** This training samples (microduck_rl's
velocity task, as is):

- forward -0.4 to 0.4 m/s, sideways -0.3 to 0.3 m/s, turn -1.0 to 1.0 rad/s, held fixed (a ramp
  to ±2.0 rad/s outpaced the robot upstream); 15 % of environments turn on the spot, and some stand;
- head angles from a small starting range, widened by the head curriculum.

A test (`test_the_reference_timeline_stays_inside_what_training_saw`) holds this folder's
`timeline.json` inside the trained walking ranges, the head inside its starting range and the body at
zero. A timeline outside it may still play well: check it on the stage, and if the user wants it
anyway, adjust the test with them.

What a walker cannot show: the grapevine's cross-behind (it sidesteps), the scuffs and touches, and
singing (an entry carries no sound). Head moves can add style within the head ranges.

## Files

| File | What it holds | Change it? |
|---|---|---|
| `timeline.json` | The performance | Yes: the user's creative part, and free to change (no training) |
| `challenge.toml` `[params]` | The three knobs below | Yes. Keep each value within its `min`/`max` (the workshop refuses others) |
| `challenge.toml` `[recipe]` | `num_envs` 4096, `max_iterations` 3000, `seed` 1 | Yes, when agreed with the user |
| `challenge.toml` `[command]` | What the policy reads in each command group; `publish` writes it into the manifest | Only with the commands' meaning |
| `env.py` | The whole environment (microduck_rl's velocity task, as the 2 m Sprint first copied it): walking and head rewards, command ranges, curricula, domain randomization, pushes, `RL_CFG` | Yes, once the knobs are not enough |
| `rewards.py` | Does not exist yet: create it for a custom reward term, then add the term to `cfg.rewards` in `env.py` | Yes |
| `check.py` | Judges a policy's performance of a timeline on the Arena's stage | Use it as is |
| `tasks.py`, `event`/`task`/`kind` in `challenge.toml` | Registration with mjlab and the Arena | No |

| Knob | Label | Default | Range | Sets in `env.py` |
|---|---|---|---|---|
| `forward_reward` | Follow the walking moves | 2.0 | 0.5–5.0 | `track_linear_velocity` weight |
| `upright_reward` | Stay upright | 2.0 | 0.5–5.0 | `upright` weight |
| `step_reward` | Lift your feet | 3.0 | 0.0–6.0 | `air_time` weight |

Do not change, or the Arena cannot play the policy or the tests fail:

- the robot (`MICRODUCK_STANDUP_ROBOT_CFG`: the walk robot with the Arena's collision set) and the
  61-number observation layout: an unused command slot stays zero-padded, never deleted;
- the contact solver at the Arena's 10/20 iterations. (Since dance1 a fall ends a training episode,
  as in every walking task: a walker cannot get up. The stage still plays on after one.)
- `experiment_name = "got_a_talent"` in `RL_CFG`: runs must land in `logs/rsl_rl/got_a_talent/`;
- every domain randomization range must contain the nominal value the Arena simulates (the Arena
  randomizes nothing but the start pose and the servos' command delay; no pushes, no noise);
- `pyproject.toml`, `uv.lock` and the other challenges' folders.

The robot model and the reward functions (`microduck_mdp`) live in the `mjlab-microduck` package
and are not edited here. Its [AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/develop/AGENTS.md)
holds the reward-design rules learned on this robot: read its "Reward design" section before
writing a reward term.

## Working on a performance

### Once per session

```bash
uv sync --group workshop
nvidia-smi              # an NVIDIA GPU here trains locally; otherwise Hugging Face Jobs
uv run hf auth whoami   # the account a job runs and bills under, and publish publishes to
```

The Arena ships Pollen's general walker, which lets you try a timeline before any training:

```bash
uv run --group workshop python -m microduck_challenges.got_a_talent.check .venv/lib/python3.12/site-packages/arena/assets/policies/BEST_alpha_walking.onnx
```

With this folder's Electric Slide it dances every section without a fall, but follows only 15-20 %
of each sideways move (sidesteps of about 0.1 m for 0.6 m commanded), so `check.py` says FAIL; back
and forward moves reach 33-45 % and the quarter turn 62-72 %. Add `--timeline <file>` to check
another timeline.

Agree with the user before the first run: the performance they want; whether their policy (at
first, the recipe trained as is) can already perform it, so that only `timeline.json` changes; and
the budget (local GPU, or Hugging Face Jobs: default hardware `l4x1`, $0.80 per hour, billed to
their account, which needs a payment method; the 3000-iteration recipe took 2 h 20).

### Editing the timeline

1. Write it with the user: what happens when, in plain words, then as keyframes.
2. Check it on the stage with the policy it will be published with:
   `uv run --group workshop python -m microduck_challenges.got_a_talent.check <policy.onnx>`
   prints a line per keyframe (posture reached and held, head error, fall) and PASS or FAIL;
   `uv run --group workshop arena score <policy.onnx> --event got-a-talent --timeline src/microduck_challenges/got_a_talent/timeline.json`
   prints the Arena's own summary (length played, fall, seconds seated and standing).
3. Show it: the workshop (`uv run --group workshop workshop got-a-talent`) edits `timeline.json` on a
   training's page and plays it with the checkpoint the user picks.
4. Commit it: `git add src/microduck_challenges/got_a_talent && git commit -m "<what the performance does>"`.

### Each training

1. **State the hypothesis** in one sentence: what the policy cannot do yet, what you change, what
   you expect to see.
2. **Change one thing**, a knob first. Combine changes only on purpose, and say so.
3. **Test**: `uv run --group workshop pytest`.
4. **Smoke test** (one to two minutes, also on a machine without a GPU):

   ```bash
   uv run train Mjlab-GotATalent-MicroDuck --gpu-ids None --env.scene.num-envs 2 \
       --agent.max-iterations 2 --agent.logger tensorboard --agent.run-name smoke
   rm -rf logs/rsl_rl/got_a_talent/*_smoke
   ```

5. **Commit** the change, its message the hypothesis:
   `git add src/microduck_challenges/got_a_talent && git commit -m "<hypothesis>"`. `publish` writes
   that commit into the policy's recipe, and refuses a run started from uncommitted changes.
6. **Launch** in the background, the output in a log under `logs/` (git-ignored). `<name>` is short,
   unique, and says the change (`head1`, `still3`):

   ```bash
   # local NVIDIA GPU
   uv run train Mjlab-GotATalent-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 3000 \
       --agent.seed 1 --agent.logger tensorboard --agent.run-name <name> > logs/<name>.log 2>&1 &
   # Hugging Face Jobs: the same, plus
   #   --hf-jobs --no-wandb --namespace <user> --ckpt-repo <user>/mjlab-gotatalent-microduck-<name>
   ```

   A local run writes `logs/rsl_rl/got_a_talent/<YYYY-MM-DD_HH-MM-SS>_<name>/`: `model_<N>.pt` every
   250 iterations and the last one, TensorBoard events, `provenance.json`. A job uploads the same
   tree to its checkpoint repo. Its log prints `[job] id: <id>`. Killing the local process only
   detaches from the job: stop it with `uv run hf jobs cancel <id> --namespace <user>`.

### Following a run

- Progress: `grep -o "Learning iteration [0-9]*/[0-9]*" logs/<name>.log | tail -1`.
- A job's checkpoints and curves: `uv run hf download <user>/mjlab-gotatalent-microduck-<name> --local-dir logs/rsl_rl`
  (the repo mirrors `logs/rsl_rl/`); its state: `uv run hf jobs inspect <id> --namespace <user>`.
- Check once per checkpoint (every 250 iterations, several minutes apart), not more often.
- Curves (`Episode_Reward/<term>` is the weighted value; penalties must read ≤ 0):

  ```bash
  uv run python -c "
  import sys; from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
  ea = E(sys.argv[1], size_guidance={'scalars': 0}).Reload()
  for tag in sys.argv[2:]: print(tag, [(e.step, round(e.value, 3)) for e in ea.Scalars(tag)][-8:])
  " logs/rsl_rl/got_a_talent/<run> Episode_Reward/track_linear_velocity Episode_Reward/track_angular_velocity Episode_Reward/head_pose_tracking Episode_Termination/fell_over
  ```

  `track_linear_velocity` and `track_angular_velocity` are the main task terms: they must keep
  growing while `fell_over` falls. Curriculum stages land at iterations 500 to 2000 (standing
  environments, head range, action-rate smoothing): a drop right at one means the pacing is wrong. A
  run that has collapsed and not recovered within about 500 iterations: tell the user and propose
  stopping it.

### Judging every checkpoint

The curves are not the performance. Judge each checkpoint on the Arena's stage with the timeline
the user will publish:

- If the user has the workshop open, it finds every run under `logs/rsl_rl/`, yours included, and
  plays the timeline with each checkpoint as it arrives. Its results:
  `logs/rsl_rl/got_a_talent/<run>/workshop/model_<N>.json` (`stage`: `fell_at`, `seated_s`,
  `standing_s`).
- Otherwise:

  ```bash
  run=logs/rsl_rl/got_a_talent/<run>
  for n in $(ls $run | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n); do
    uv run publish --run $run --checkpoint $n --repo local/microduck-eval --dry-run --allow-dirty >/dev/null 2>&1 &&
    echo "checkpoint $n: $(uv run --group workshop python -m microduck_challenges.got_a_talent.check publish-eval/policy.onnx | tail -1)"
  done
  ```

  `--dry-run` exports exactly as the real publish (the observation normalizer baked in) into
  `publish-eval/`, and publishes nothing.

Pick a checkpoint that passes, then the one that follows the largest share of its moves (the
verdict line) without a fall. The Arena plays one start, without training's randomization: a
checkpoint whose neighbours pass too is the safer pick. The best checkpoint is often not the last. GPU training varies a lot
from run to run: before building on a surprising result, run the same commit with another
`--agent.seed`.

### Reporting

After each run, tell the user in a few lines:

```
run dance2 (commit 1a2b3c4: forward_reward 2.0 → 3.0, to follow the sidesteps more closely)
best: checkpoint 2250, PASS, smallest share of a move 41 %, no fall   (Pollen's walker FAIL, 15 %)
others: 1500 FAIL (fell at 12.3 s) · 2000 PASS 33 % · 2999 PASS 38 %
next: <your proposal and why>
```

### Publishing, on the user's yes

Publishing makes the policy and its performance public on their Hugging Face account and enters it
on the Arena, which plays it and prints the page of its result. **Pass the timeline**, or the entry
stands still for 20 s:

```bash
uv run publish --run logs/rsl_rl/got_a_talent/<run> --checkpoint <N> --repo <user>/microduck-<name> \
    --no-private --timeline src/microduck_challenges/got_a_talent/timeline.json --enter
```

`--no-private` because the Arena reads public repos only; add `--livery <colour>` for its colour.
For the policy's code link to open, push the commit to the user's fork first (`origin` must be their
fork, never `pollen-robotics`; the workshop's home sets that up).

## What is known about this challenge

- **The timeline keeps the time, not the policy.** Before this format, the challenge asked the
  policy to time a sit-and-stand routine on its own under a constant command: ten runs failed. A
  clock kept in the policy's own joints washes out under PPO's exploration noise within seconds.
  Any new move should be triggered by a command, never by the policy counting.
- **dance1 (2026-10-19) swapped the sit & stand task for a walker.** The starting recipe was
  microduck_rl's sit & stand task (sits in about 1 s, rises in under 0.5 s, at the Arena's 10/20
  solver); it is in this folder's git history. The walker is microduck_rl's velocity task with
  the Arena's collision set and solver.
- **Judge the exported, deterministic policy.** One failed attempt completed its moves only with
  exploration noise on; deterministic, it never moved. `check.py` and the workshop judge the export.
- **Reward traps met on this robot**, worth knowing before adding a move: a state that earns the
  same before and after a move (standing still) means the move is never started; a reward window
  that closes before the move completes makes the policy unlearn it; a "rise" that was really a
  lean back at 57°, just under the 60° fall threshold.
- **A new kind of move needs a command for it.** The 13 command numbers are fixed (the real robot
  feeds the same 61 observations). Free slots: `twist[1]`, `twist[2]` and `body` (6). A new move
  reads one of them, is trained on it, and `[command]` in `challenge.toml` says what it means.

## The competition

- The Arena is the Hugging Face Space
  [pollen-robotics/microduck-arena](https://huggingface.co/spaces/pollen-robotics/microduck-arena).
  The board ranks performances by votes over the last seven days. Read it live:
  `curl -s https://pollen-robotics-microduck-arena.hf.space/api/events/got-a-talent/leaderboard`
  (each row: `user`, `policy_name`, `votes`, `shown`).
- Player points: +100 for a first publication in a challenge, +10 per publication (the first 3 per
  UTC day, all challenges together), +250/+150/+100 while holding first/second/third place (a
  voted entry needs an active vote). Receiving votes earns no personal-best bonus. Voters earn 2
  points per accepted vote, 20 votes per UTC day.
- The challenge's page on the Arena has the same rules in plain words, and the **Train my own robot**
  button that opens the workshop.
