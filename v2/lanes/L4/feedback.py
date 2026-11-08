#!/usr/bin/env python3
"""L4 feedback command: takes <=25 annotated items (jsonl rows {"text":..., "label":...}) and applies the closed-form diffuse edit to the o_proj of the last layers
(no optimiser, no new tensor, no side memory, nothing in the prompt). Implementation = v2/rival3/fb.py `edit()` (teacher-force the corrected answer, residual target
dx = rms*(gap+m)*u/(u.u), ridge solve with calibration-covariance protection, layer l delivers 1/(layers left) of the remaining dx).
Measured through the harness (v2/rival3/fb.py main loop, Modal app tenpct-rival3); this wrapper is a thin CLI over it and is NOT yet runnable standalone against a saved checkpoint."""
import sys; print(__doc__); sys.exit(0)
