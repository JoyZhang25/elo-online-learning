# Elo as Constant-Step SGD

**Global tail geometry, invariant fluctuations, adaptation, and walk-forward prediction**

Elo is a constant-step stochastic-gradient algorithm for sequential pairwise
outcomes. This repository follows one question from theory to data:

> What does the constant step do when the objective has nonquadratic tails,
> but quadratic geometry near its optimum?

The answer has three parts. In a stationary Elo model, the rating error has an
invariant distribution whose scale is approximately $\sqrt{K}$ and whose
standardized shape is close to Gaussian. In a changing environment, the same
constant $K$ controls the trade-off between estimation noise and adaptation
speed. On real games, this mechanism can be evaluated without look-ahead by
predicting each outcome before updating the ratings.

This is an applied numerical illustration of the locally quadratic regime in
[*Scaling Limits of Constant-Stepsize SGD at Flat Minima*](https://arxiv.org/abs/2607.16384),
not a reproduction of the paper's genuinely flat higher-order regime.

## Results at a glance

| Question | Design | Main result |
|---|---|---|
| Do global linear tails prevent a local Gaussian regime? | Fixed-skill Elo against known-rating opponents; $K\in\{2,4,8,16,32\}$ | Stationary error variance is nearly proportional to $K$: variance/$K$ ranges from 84.0 to 89.7. Standardized QQ curves are close to Gaussian. |
| Why keep a constant step when skills can change? | 24-player drift simulation and a 240-point skill jump | Small $K$ is most stable when skill is fixed; larger $K$ tracks drift and breaks faster. $K=64$ reaches 90% of the jump in 24 matches, versus 237 for $K=8$, but remains noisier. |
| Does online updating help on real games? | Strictly chronological evaluation on 4,073 Lichess blitz games | Validation selects $K=64$. Test log loss is 0.5974, versus 0.6050 for frozen initial ratings and 0.5996 for Lichess pre-game ratings. The improvement is modest and not uniform across every metric. |

## 1. The model and the paper connection

Let $x=r_i-r_j$ be the rating difference and let

$$
c=\frac{\log 10}{400}.
$$

The Elo expected score of player $i$ against player $j$ is the logistic model

$$
p(x)=\frac{1}{1+e^{-cx}}
=\frac{1}{1+10^{-x/400}}.
$$

For an observed score $Y_t\in\{0,1/2,1\}$, the standard update is

$$
r_{i,t+1}=r_{i,t}+K\bigl(Y_t-p_t\bigr),
\qquad
r_{j,t+1}=r_{j,t}-K\bigl(Y_t-p_t\bigr).
$$

For binary outcomes, the gradient of logistic loss contains the fixed factor
$c$. After absorbing that factor into the learning rate, $K$ is the effective
SGD step size. With draws, $p_t$ is interpreted as expected score rather than a
literal win probability.

### Linear tails, quadratic local geometry

Suppose the true score probability is $q\in(0,1)$. The one-dimensional
population logistic loss is

$$
L(x)=\log(1+e^{cx})-qcx.
$$

It has asymptotically linear tails:

$$
L(x)\sim c(1-q)x \quad (x\to+\infty),
\qquad
L(x)\sim -cqx \quad (x\to-\infty).
$$

Thus the global objective is not quadratic. At its optimum $x^\star$, however,
$p(x^\star)=q$ and

$$
L''(x^\star)=c^2q(1-q)>0.
$$

The local geometry is therefore quadratic. For several players, the Hessian is
a weighted graph Laplacian. The exact identity

$$
L(r+a\mathbf 1)=L(r)
$$

is an additive-identifiability symmetry, not a higher-order flat minimum. Once
the common-shift direction is removed, a connected population comparison graph
has positive curvature on the identifiable rating subspace.

This separates three ideas that are easy to conflate:

- the loss has **linear tails** far from the optimum;
- rating levels have one **unidentifiable shift direction**;
- on the identifiable subspace, the optimum has **quadratic local geometry**.

The paper predicts that, under its regularity and recurrence conditions, this
locally quadratic case has $\sqrt{K}$-scale Gaussian invariant fluctuations as
$K\downarrow0$ (up to the fixed conversion between $K$ and the paper's step
parameter). Higher-order local flatness changes both the scale and the limiting
law; changing only the global tail shape does not.

## 2. Controlled evidence

### 2.1 Invariant fluctuations under fixed skill

One focal player with fixed latent rating repeatedly faces opponents whose
ratings are known. Outcomes are generated from the same logistic model. Every
candidate $K$ sees the same opponent sequence and outcomes; after 40,000 burn-in
games, the experiment keeps one rating error every 20 games.

![Tail geometry, variance scaling, and Gaussian diagnostics](docs/figures/theory_bridge.png)

The three panels make one argument:

1. the population loss is globally linear-tailed but locally quadratic;
2. stationary error variance is approximately proportional to $K$, so the
   fluctuation scale is approximately $\sqrt{K}$;
3. standardized stationary errors lie close to a Gaussian QQ reference line.

These are finite-$K$ diagnostics, not a numerical proof of an asymptotic
theorem. The saved summary also reports the empirical mean, skewness, and excess
kurtosis for every $K$.

### 2.2 Why a constant step is useful in a changing system

The second experiment moves outside the stationary theorem and asks why an
online rating system would retain a nonvanishing step. It simulates chronological
matches among 24 players whose latent skills are fixed or follow centered random
walks. A separate change-point experiment tracks one focal player after a
240-point skill jump against a stream of known-rating opponents.

![Stability-adaptation trade-off and change-point tracking](docs/figures/simulation_tradeoff.png)

When skill is stationary, $K=2$ has rating RMSE 12.3, compared with 78.7 for
$K=64$. Under the largest simulated drift, the best value moves to $K=8$ and
reduces RMSE from 64.2 at $K=2$ to 42.2. After the abrupt jump, large $K$ reacts
faster but retains more post-change variation. A decaying step eventually
stabilizes, but also loses its original adaptation speed.

The two controlled experiments play different roles: the fixed-skill experiment
connects to invariant-law theory, while drift and change points explain the
practical reason to use a constant step.

## 3. Walk-forward prediction on real games

The empirical sample is the completed
[Lichess Daily Blitz Arena on 18 September 2025](https://lichess.org/tournament/NQzyuRkI).
The official export contains 4,225 records. The pipeline retains 4,073 rated
standard games with two identified players and a completed result, then:

1. parses NDJSON into a typed pandas table;
2. removes ineligible records and duplicate game IDs;
3. sorts by creation time rather than API return order;
4. initializes each new player from the rating visible before that player's
   first sampled game;
5. records each probability before using the result to update Elo.

The chronological split is fixed in advance:

- first 10%: state warm-up;
- next 50%: select $K$ and a white-advantage offset by validation log loss;
- final 40%: locked test comparison.

No test result is used to select a hyperparameter.

![Walk-forward model selection, calibration, and cumulative loss](docs/figures/real_data_evaluation.png)

### Held-out results

| Model | Games | Log loss | Brier score | Calibration error |
|---|---:|---:|---:|---:|
| Constant-step Elo | 1,630 | 0.5974 | 0.1947 | 0.0669 |
| Lichess pre-game ratings | 1,630 | 0.5996 | 0.1946 | 0.0710 |
| Frozen initial ratings | 1,630 | 0.6050 | 0.1959 | 0.0790 |
| No-skill 50% | 1,630 | 0.6931 | 0.2419 | 0.0304 |

Within this tournament, online updating improves log loss relative to freezing
each player's initial rating and is competitive with the platform benchmark.
The platform probabilities have a slightly lower Brier score, so the evidence
does not support a claim of uniform dominance. The 50% model's small calibration
error is also not evidence of useful prediction: it has the worst proper scores
because it contains no ranking information.

## Reproduce the project

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v

# Controlled theory and adaptation experiments
python scripts/run_simulations.py

# Fixed real-data study; downloads the official export if cache is absent
python scripts/run_real_data.py

# Rebuild every committed figure and table
python scripts/build_all.py
```

The controlled simulations use fixed random seeds. The real-data outputs are
deterministic conditional on the fixed tournament export.

## Repository map

```text
elo_online/
  model.py          # Logistic expected score and stateful Elo updates
  simulation.py     # Invariant, drift, and change-point experiments
  data.py           # Official Lichess download, validation, and cleaning
  evaluation.py     # Chronological selection and walk-forward prediction
  metrics.py        # Log loss, Brier score, and calibration
  reporting.py      # Figures, anonymized tables, and reproducible outputs
scripts/            # One-command experiment entry points
tests/              # Update, simulation, ordering, parsing, and metric tests
docs/figures/       # Committed visual results
docs/tables/        # Committed tidy numerical outputs
```

## Scope

- The invariant plots are finite-step diagnostics, not a proof of the scaling
  theorem or an experiment in the paper's higher-order flat regime.
- The empirical study is online probabilistic prediction, not a chess engine.
- The real sample covers one two-hour arena and does not establish a universal
  $K$ or a production rating system.
- External pre-game ratings provide a defensible cold start, so this is not a
  from-scratch skill model.
- Arena pairings and repeated players create dependence. Scores are descriptive
  out-of-sample summaries, not independent-observation hypothesis tests.
- Draws are encoded as fractional outcomes; a dedicated draw model is outside
  the scope of this compact project.

## Data and license

Lichess database exports are released under
[CC0](https://database.lichess.org/). Raw responses are cached locally and are
not redistributed. The committed prediction table irreversibly hashes player
identifiers.

Project code is released under the [MIT License](LICENSE). That license does
not assert ownership over Lichess data or the linked research paper.
