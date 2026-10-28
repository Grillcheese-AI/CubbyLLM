"""validation/pod -- the rented-card runner (order 0045): the inputs built at home and uploaded to a private repo
(`prepare_inputs.py`), the pod's one command (`setup.sh` -> `run_pod.py`), its phases (`pod_phases.py`) and the
remote it resumes from (`pod_remote.py`). Standalone scripts; nothing in `cubbyllm/` imports them.

Wired: STANDALONE (measurement tooling).
"""
__wiring__ = "STANDALONE"
