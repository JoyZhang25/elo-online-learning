# Experiment Notes

These notes record the choices behind the figures and tables in this
repository. They are intentionally more practical than theoretical: the main
question is what the code is testing and what the results should not be used
to claim.

## 1. Invariant-Fluctuation Simulation

File: `elo_online/simulation.py`

The focal player has true rating 1500 and repeatedly plays opponents with
known ratings drawn around 1500. Outcomes are generated from the same Elo
expected-score model used by the estimator. For each tested value of `K`, the
opponent sequence and outcomes are held fixed.

Default settings:

- `K` grid: 2, 4, 8, 16, 32
- games: 200,000
- burn-in: 40,000
- retained sample: every 20th post-burn-in update
- random seed: 20260819

The output table is `docs/tables/invariant_fluctuation_summary.csv`. The
diagnostic I check is whether `variance / K` stays roughly stable and whether
standardized errors are close to Gaussian in a QQ plot.

This experiment is deliberately narrow. It illustrates the locally quadratic
case of the logistic objective. It does not test the higher-order flat-minimum
regime studied in flat-minima SGD theory.

## 2. Stability and Drift

File: `elo_online/simulation.py`

This simulation creates a small pool of 24 players. Their true ratings are
either fixed or slowly drifting. The estimator sees only game outcomes and
updates the two players in each observed match.

Default settings:

- `K` grid: 2, 4, 8, 16, 32, 64
- drift standard deviations: 0, 0.6, 1.0 Elo points per event
- games: 24,000
- burn-in fraction: 50 percent
- random seed: 20260818

The reported RMSE aligns the estimated and true rating vectors by their mean
before computing error. This removes the irrelevant common-shift degree of
freedom in the Elo ratings.

The separate change-point experiment gives one focal player a 240-point jump
halfway through the sequence. I record post-change RMSE and the number of
matches needed to reach 90 percent of the jump.

These simulations are useful for understanding the role of `K`, but the
pairing process is synthetic. The numbers should not be read as recommended
chess-rating parameters.

## 3. Lichess Walk-Forward Evaluation

Files: `elo_online/data.py`, `elo_online/evaluation.py`

The real-data experiment uses one completed public Lichess tournament:

- tournament id: `NQzyuRkI`
- tournament page: <https://lichess.org/tournament/NQzyuRkI>
- official export endpoint: `https://lichess.org/api/tournament/NQzyuRkI/games`
- date: 18 September 2025

The cleaning step keeps rated standard games with both player ids and both
platform ratings available. It drops duplicate game ids and sorts by
`created_at`, then by `game_id` as a stable tie-breaker.

The evaluation is chronological:

- 10 percent warm-up block;
- 50 percent validation block for `K` and white-advantage selection;
- 40 percent test block for final reporting.

Every candidate model runs forward from the first game. Predictions are made
before the corresponding game outcome is used for an update. This matters
because even a small accidental look-ahead would make the online comparison
too optimistic.

## 4. Baselines

The test table compares four predictors:

- constant-step Elo with `K` and white advantage selected on validation;
- frozen initial ratings, where each player keeps the first visible platform
  rating used to initialize them;
- Lichess pre-game ratings, evaluated through the same expected-score link;
- a constant 50 percent predictor.

The Lichess benchmark is not exactly the platform's internal win model. It is
only the public pre-game rating difference passed through the standard Elo
expected-score formula.

## 5. Current Limitations

- The real-data result comes from one tournament. It is a useful fixed sample,
  not a broad empirical study of online Elo.
- Player appearances are highly dependent because tournaments contain repeated
  players and uneven pairing patterns.
- Cold-start ratings come from the platform ratings visible in the export.
  This is practical for prediction, but it means the experiment is not learning
  every player's rating from scratch.
- Draws are encoded as score 0.5 and evaluated by the same binary proper-score
  formulas. That is natural for Elo expected score, but it is not a separate
  three-outcome model.
- The simulations are model-correct by construction. They check behavior
  under controlled assumptions, not robustness to model misspecification.

## 6. Possible Next Steps

- Repeat the walk-forward evaluation on several tournaments and time controls.
- Compare constant `K` with a decaying step and with player-specific adaptive
  steps.
- Add uncertainty intervals using a block bootstrap over chronological chunks.
- Separate white advantage from tournament-specific player effects.
- Test a three-outcome model for win/draw/loss instead of treating draws only
  as expected score 0.5.
