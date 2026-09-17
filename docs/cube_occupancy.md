# Cube occupancy after the corner remap

120 frames per situation, each held steady from a fresh body. Signals are the ODE's own inputs, not emotion words - the question is what each situation PRODUCES.

| situation | corners visited |
|---|---|
| idle, nothing happening | **neutral** 114, **surprise** 3, **disgust** 3 |
| exploring, new ground | **anger** 120 |
| pellet eaten | **anger** 120 |
| ghost closing in | **distress** 120 |
| ghost near, still exploring | **distress** 69, **anger** 51 |
| caught - hurt, no prospect | **distress** 120 |
| stuck, nothing works | **distress** 106, **neutral** 14 |
| level cleared | **anger** 120 |
| caught + pain injected | **distress** 120 |

## Reachability

| corner | coord | frames | reachable |
|---|---|---:|---|
| spent | (0, 0, 0) | 0 | **NO** |
| distress | (0, 0, 1) | 535 | yes |
| fear | (0, 1, 0) | 0 | **NO** |
| anger | (0, 1, 1) | 411 | yes |
| disgust | (1, 0, 0) | 3 | yes |
| surprise | (1, 0, 1) | 3 | yes |
| joy | (1, 1, 0) | 0 | **NO** |
| interest | (1, 1, 1) | 0 | **NO** |
| neutral | centre | 128 | - |

## Reading this

**Unreachable: spent, fear, joy, interest.** A corner no situation can produce is a corner the agent can never be in - the same class of bug as the pre-scaling cube, where noradrenaline's band made half the vertices impossible by construction.

`fear` specifically is the one to argue about. Lövheim puts it at low 5-HT / HIGH DA / low NE, and this ODE subtracts `threat * 0.45` from dopamine - so threat pushes the body away from the fear corner and toward distress. Either the coupling is too strong, or this ODE's threat response genuinely is distress rather than fear. That is a modelling decision, not a bug to patch quietly.
