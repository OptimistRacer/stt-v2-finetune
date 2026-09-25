"""Keep the GPU at a steady, moderate temperature during training.

This machine has crashed outright several times during GPU work (driver fault ->
unclean reboot; see scripts/run_day3_train.py), and under full training load its
RTX 5090 is loud and runs at the edge: 85 C with fans at 97%, climbing ~5 C per second,
reaching 88 C before a simple check-every-few-steps cutoff could react (it throttles
around 90 C).

Two layers:

1. Pacing (the main one). After each step, read the temperature; if it's above
   `target_c`, rest for `gain_s_per_c` seconds per degree over, capped at `max_rest_s`.
   The card then settles *near* the target instead of racing to the limit and stopping
   -- which is also what keeps it quiet, since the fans follow temperature, and a
   steady temperature means steady fans rather than repeated ramps to full speed.

2. Hard stop (safety net). If the temperature still reaches `pause_at_c`, stop until it
   falls to `resume_at_c`.

The cost is time: pacing trades speed for a cooler, quieter, safer run.

Reads temperature with nvidia-smi rather than adding a dependency. Fails open: if the
temperature can't be read, training continues rather than stalling forever.
"""
import subprocess
import time

import torch
from transformers import TrainerCallback


def gpu_temperature(index: int = 0) -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits", "-i", str(index)],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class ThermalGuard(TrainerCallback):
    def __init__(self, target_c: int | None = None, gain_s_per_c: float = 0.25, max_rest_s: float = 4.0,
                 pause_at_c: int = 80, resume_at_c: int = 70, check_every_steps: int = 1,
                 poll_s: float = 1.0, max_pause_s: float = 600.0, log_every_steps: int = 100):
        if resume_at_c >= pause_at_c:
            raise ValueError("resume_at_c must be below pause_at_c, or training would never pause")
        if target_c is not None and target_c >= pause_at_c:
            raise ValueError("target_c must be below pause_at_c, or pacing never engages")
        self.target_c = target_c
        self.gain_s_per_c = gain_s_per_c
        self.max_rest_s = max_rest_s
        self.pause_at_c = pause_at_c
        self.resume_at_c = resume_at_c
        self.check_every_steps = check_every_steps
        self.poll_s = poll_s
        self.max_pause_s = max_pause_s
        self.log_every_steps = log_every_steps

        self.rested_s = 0.0
        self.hard_pauses = 0
        self.paused_s = 0.0
        self.peak_c = 0
        self.temps = []

    def _regulate(self):
        temp = gpu_temperature()
        if temp is None:
            return
        self.peak_c = max(self.peak_c, temp)
        self.temps.append(temp)

        if temp >= self.pause_at_c:
            self._hard_pause(temp)
        elif self.target_c is not None and temp > self.target_c:
            rest = min(self.max_rest_s, self.gain_s_per_c * (temp - self.target_c))
            torch.cuda.synchronize()
            time.sleep(rest)
            self.rested_s += rest

    def on_substep_end(self, args, state, control, **kwargs):
        # With gradient accumulation, one optimizer step is several micro-batches run
        # back to back. Checking only at on_step_end let the card climb past the hard
        # stop within a single step (236 hard pauses in 250 steps, pacing barely used).
        # Checking between micro-batches makes each uninterrupted burst a fraction as
        # long, so pacing can hold the temperature instead of the safety net doing it.
        self._regulate()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.check_every_steps:
            return
        self._regulate()

        if state.global_step % self.log_every_steps == 0:
            recent = self.temps[-self.log_every_steps:]
            print(f"[thermal] step {state.global_step}: avg {sum(recent) / len(recent):.0f}C, "
                  f"peak so far {self.peak_c}C | rested {self.rested_s / 60:.1f} min, "
                  f"hard pauses {self.hard_pauses}", flush=True)

    def _hard_pause(self, temp):
        torch.cuda.synchronize()
        start = time.time()
        while temp is not None and temp > self.resume_at_c and time.time() - start < self.max_pause_s:
            time.sleep(self.poll_s)
            temp = gpu_temperature()
        self.hard_pauses += 1
        self.paused_s += time.time() - start

    def on_train_end(self, args, state, control, **kwargs):
        avg = sum(self.temps) / len(self.temps) if self.temps else float("nan")
        print(f"[thermal] run finished: average {avg:.0f}C, peak {self.peak_c}C | "
              f"paced rest {self.rested_s / 60:.1f} min, {self.hard_pauses} hard pauses "
              f"({self.paused_s / 60:.1f} min)", flush=True)
