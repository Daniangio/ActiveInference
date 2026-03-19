# Surprise-modulated predictive plasticity toy

This project is a small research toy for the idea:

> a circuit learns to predict its near future, and plasticity is amplified when the current prediction error is unexpectedly larger than its recent baseline.

It is **not** a proof of active inference or a biophysical neuron model. It is a compact sandbox to test a local learning rule.

## What is in the toy

The circuit contains:

- an **excitatory pathway** from recent inputs
- an **inhibitory pathway** from recent inputs
- a small **memory unit**
- an **autoregressive self-term** `alpha`
- a **surprise gate** driven by a short-vs-long timescale mismatch in absolute prediction error

Prediction is:

\[
v_t = E_t + \kappa m_t - I_t + \alpha y_{t-1} + b
\qquad
p_t = \sigma(v_t)
\]

with

- `E_t = w_exc dot x_t`
- `I_t = beta * sigmoid(w_inh dot x_t + b_inh)`
- `m_t = tanh(w_mem dot x_t + a_mem * m_{t-1} + b_mem)`

Plasticity uses the Bernoulli prediction error:

\[
e_t = y_t - p_t
\]

and a thresholded gate:

\[
novelty_t = \frac{\max(0, EMA_fast(|e|) - EMA_slow(|e|) - margin)}{EMA_slow(|e|)+\epsilon}
\]

\[
g_t = gate\_baseline + gate\_strength \cdot clip(novelty_t, 0, gate\_cap)
\]

The idea is that **persistent error bursts** should amplify learning more than isolated noisy mistakes.

## Why the gate is not raw E/I imbalance

In the earlier toy, a gate based directly on `|E - I| / (E + I)` did not reliably track rule switches. That quantity mainly reflected the current parameterization rather than true surprise.

In this version:

- E and I are still modeled and visualized
- but the **learning gate** is based on error timescale mismatch
- E/I imbalance is treated as an observable internal variable, not as the sole plasticity controller

This separation is cleaner scientifically.

## About `alpha`

Yes: if `alpha` is part of the predictor, it should usually be learned. In this toy, `alpha` is updated online. Still, `alpha` is only a crude memory term. The small recurrent memory unit `m_t` is a slightly better approximation of internal state.

## Files

- `run_experiment.py` runs the numerical toy and saves plots
- `visualize_pygame.py` launches an interactive visualization
- `src/surprise_toy/data.py` sequence generation
- `src/surprise_toy/model.py` circuit and plasticity rule
- `src/surprise_toy/experiment.py` experiment utilities
- `src/surprise_toy/plotting.py` matplotlib figures
- `src/surprise_toy/pygame_app.py` live simulation

## Install

```bash
pip install -r requirements.txt
```

## Run

Generate plots:

```bash
python run_experiment.py --seed 0 --out-prefix outputs/demo
```

Launch the live visualization:

```bash
python visualize_pygame.py --seed 0
```

## Controls in pygame

- `SPACE`: pause / resume
- `UP` / `DOWN`: increase / decrease speed
- `T`: toggle online learning on/off
- `R`: reset model and replay
- `ESC`: quit

## What to look for

This toy is successful if, near rule switches,

- the short-vs-long error mismatch rises
- the gate transiently increases
- the model adapts faster than a plain learner in at least some settings

That last point is **not guaranteed**. This is a research sandbox, not a pre-proven mechanism.
