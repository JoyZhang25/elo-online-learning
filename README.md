# Elo as Constant-Step SGD

**Invariant fluctuations, adaptation, and walk-forward prediction**

Elo is a constant-step stochastic-gradient algorithm for sequential pairwise
outcomes. This repository follows one question from theory to data:

> What does a constant step do when the objective has linear tails but
> quadratic geometry near its optimum?

The project gives three answers:

- **Stationary regime:** rating-error variance is approximately proportional
  to the step size $K$, and standardized errors are close to Gaussian.

- **Changing skills:** larger $K$ adapts faster to drift and structural
  breaks, at the cost of greater stationary variation.

- **Real games:** a leakage-free walk-forward study shows a modest improvement
  over frozen ratings on 4,073 Lichess blitz games.

The first result is a numerical illustration of the locally quadratic regime
in [*Scaling Limits of Constant-Stepsize SGD at Flat Minima*](https://arxiv.org/abs/2607.16384).
It is not a reproduction of the paper's higher-order flat regime.

## 1. From Elo to constant-step SGD

Let $x=r_i-r_j$ and $c=\log(10)/400$. The Elo expected score is

$$
p(x)=\frac{1}{1+e^{-cx}}=\frac{1}{1+10^{-x/400}}.
$$

After observing $Y_t\in\{0,1/2,1\}$, the ratings are updated by

$$
r_{i,t+1}=r_{i,t}+K(Y_t-p_t),
\qquad
r_{j,t+1}=r_{j,t}-K(Y_t-p_t).
$$

For binary outcomes, this is a stochastic-gradient step on logistic loss after
the fixed factor $c$ is absorbed into the learning rate. With draws, $p_t$
is interpreted as expected score rather than literal win probability.

### Why the paper applies

If the true score probability is $q\in(0,1)$, the population logistic loss is

$$
L(x)=\log(1+e^{cx})-qcx.
$$

Its tails are linear: $L(x)\sim c(1-q)x$ as $x\to+\infty$ and
$L(x)\sim-cqx$ as $x\to-\infty$. At the optimum $x^\star$, however,

$$
L''(x^\star)=c^2q(1-q)>0.
$$

Thus Elo is globally nonquadratic but locally belongs to the paper's $m=2$
regime. After removing the common rating-shift ambiguity, the small-step theory
predicts Gaussian invariant fluctuations on the $\sqrt{K}$ scale. The tail
geometry is not what determines this local limiting law.

## 2. Controlled experiments

### 2.1 Stationary invariant fluctuations

A focal player with fixed latent rating repeatedly faces known-rating
opponents. Every candidate $K$ sees the same opponents and outcomes. After
burn-in, the experiment records the focal player's stationary rating error.

![Tail geometry, variance scaling, and Gaussian diagnostics](docs/figures/theory_bridge.png)

Across $K=2,4,8,16,32$, the empirical ratio of variance to $K$ stays between
84.0 and 89.7, while the standardized QQ curves remain close to the Gaussian
reference line. These are finite-step diagnostics, not a proof of the
asymptotic theorem.

### 2.2 Stability versus adaptation

The next experiment asks why an online rating system would keep a nonvanishing
step. It simulates 24 players with fixed or drifting skills, then gives one
focal player a sudden 240-point skill increase.

![Stability-adaptation trade-off and change-point tracking](docs/figures/simulation_tradeoff.png)

With stationary skill, $K=2$ has rating RMSE 12.3, versus 78.7 for $K=64$.
Under the largest drift, the best value moves to $K=8$. After the abrupt
jump, $K=64$ reaches 90% of the change in 24 matches, versus 237 for $K=8$,
but its post-change path is noisier. The same step size therefore controls a
precision-adaptation trade-off.

## 3. Walk-forward prediction on real games

The empirical sample is the completed
[Lichess Daily Blitz Arena on 18 September 2025](https://lichess.org/tournament/NQzyuRkI).
The official export contains 4,225 records; filtering leaves 4,073 rated
standard games. The pipeline:

- removes ineligible and duplicate games, then sorts by creation time;

- initializes new players from their visible pre-game ratings and records each
  prediction before updating the model;

- uses the first 10% for state warm-up, the next 50% to select $K$ and white
  advantage, and the final 40% as a locked test set.

No test outcome is used for model selection.

![Walk-forward model selection, calibration, and cumulative loss](docs/figures/real_data_evaluation.png)

### Held-out results

All scores below use the same 1,630 test games.

| Model | Log loss | Brier score |
|---|---:|---:|
| Constant-step Elo | 0.5974 | 0.1947 |
| Lichess pre-game ratings | 0.5996 | 0.1946 |
| Frozen initial ratings | 0.6050 | 0.1959 |
| No-skill 50% | 0.6931 | 0.2419 |

Within this tournament, online updating improves log loss relative to frozen
ratings and is competitive with the Lichess benchmark. The platform benchmark
has a slightly lower Brier score, so the result is evidence of useful updating,
not uniform dominance.

## Reproduce

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v

python scripts/run_simulations.py
python scripts/run_real_data.py
python scripts/build_all.py
```

The simulations use fixed random seeds. Real-data outputs are deterministic
conditional on the fixed tournament export.

## Code map

- `model.py`: expected scores and stateful Elo updates.
- `simulation.py`: invariant, drift, and change-point experiments.
- `data.py` and `evaluation.py`: cleaning, chronological selection, and
  walk-forward prediction.
- `metrics.py` and `reporting.py`: scoring rules, figures, and auditable tables.
- `tests/`: update, parsing, chronology, metric, and reproducibility checks.

## Limitations

- The invariant experiment illustrates the paper's $m=2$ regime; it does not
  reproduce the higher-order flat case.
- The empirical evidence comes from one tournament and uses external ratings
  for cold starts, so it does not establish a universal $K$.
- Pairings and repeated players are dependent. Reported test scores are
  descriptive out-of-sample summaries rather than independent-observation
  hypothesis tests.

## Data and license

Lichess database exports are released under
[CC0](https://database.lichess.org/). Raw responses are cached locally and not
redistributed; committed player identifiers are irreversibly hashed. Project
code is released under the [MIT License](LICENSE).
