# Real Robot Test Protocol

## Stage 1: Offline Replay Analysis (20 Segments)

Run:

python3 /home/icrlab/tactile_work_Wy/lerobot/real_test/scripts/offline_replay_check.py \
  --config /home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json

Pass criteria:

- 20 segments generated.
- No NaN/Inf in action stream.
- No large discontinuities in xyz or rot6d trajectory.
- Out-of-bound ratio should be close to 0 before safety and exactly 0 after safety simulation.

## Stage 2: Dry-Run Real-Time Loop At 10Hz

Run with dry run adapter first:

python3 /home/icrlab/tactile_work_Wy/lerobot/real_test/scripts/realtime_inference.py \
  --config /home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json \
  --dry-run

Pass criteria:

- Mean loop frequency near 10Hz.
- Overrun ratio is low and stable.
- Latency log is generated.
- Emergency stop file can stop loop: touch /tmp/lerobot_estop

## Stage 3: Empty-Load Real Robot Test (10 Rounds)

Run protocol logger:

python3 /home/icrlab/tactile_work_Wy/lerobot/real_test/scripts/protocol_runner.py \
  --config /home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json

Execution rule:

- Run 10 rounds with no object and low speed.
- Record success/fail/estop and notes for each round.

## Stage 4: Simplified Task Test

Only proceed if empty-load stage is stable.

- Use simplified scene and reduced disturbance.
- Run configured number of rounds.
- Continue recording results in the same jsonl log.

## Mandatory Safety Rule

- Emergency stop must be physically reachable.
- If any abnormal motion appears, stop immediately and inspect logs before retry.
