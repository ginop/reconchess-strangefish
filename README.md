# StrangeFish

StrangeFish is a bot created to play reconnaissance-blind-chess (RBC).
It played in and won the NeurIPS 2019 RBC tournament!
For more information about RBC and the tournament, see
https://rbc.jhuapl.edu/ and https://rbc.jhuapl.edu/tournaments/26.

Active development of successor bots is not public, although some 
updates have been made here for compatibility. Feel free to ask 
questions or raise issues on this repo as needed.

## Disclaimer

An author of this code is an employee of the RBC competition host
institution, The Johns Hopkins University Applied Physics Laboratory
(JHU/APL), but was not at the time of the competition. JHU/APL has
promoted code availability for this and other RBC bots to assist new
participants.  JHU/APL has not endorsed this code.

## Overview

StrangeFish is separated into two parts: the primary component
manages game-state information throughout the game,
and the secondary component makes decisions for sensing and
moving based on that available information.
Because of this separation, it should be simple for users to
implement their own strategies to use within StrangeFish.

StrangeFish maintains a set of all possible board states
by expanding each possible board
into a new set for each possible opponent's move each turn.
The set of possible board states is reduced by
comparing hypothetical observations to in-game results for
sensing, moving, and the opponent's move's capture square.
Expansion and filtering of the board set are parallelized,
and set expansion is begun during the opponent's turn
(when possible).

For the NeurIPS 2019 tournament, we submitted StrangeFish
using multiprocessing_strategies.py for decision making
with the parameters tuned by config.yml.

The sensing strategy does calculate and consider the expected
board set size reduction, but the primary influence on
sensing decisions is the expected impact on the following
move choice. Before sensing, the bot calculates move scores
for the full board set (or for as large of a random sample
as time allows) and compares those to scores calculated for
each sub-set of boards that a sensing result would produce.
The multiple potential sense results for each square are
averaged, with weighting based on a (simplistic) prediction
of each board's probability of being the true board state.
The sense choice is the square with the greatest expected
impact on the same turn's move decision.

The sensing decision is therefore coupled with the move strategy.
In this case, that strategy is a flexible combination
of the best-case, worst-case, and expected (weighted-average)
outcomes for each available move across the board set.
The relative contributions of the mean, min, and max scores
are tunable. For NeurIPS 2019, StrangeFish considered only
the expected and worst-case move results; move scores were
90% mean and 10% min. The scores for each individual move
were calculated using StockFish when possible, with a set
of heuristics for evaluating board states unique to RBC
such as staying or moving into check or actually capturing
the opponent's king. The moving strategy also contributes
to the board set size maintenance by awarding additional
points to moves which are expected to eliminate possibilities
(for example, a sliding move which may travel through
potentially-occupied squares).

## Setup

We use conda to manage our execution environment.
```
conda env create --file environment.yml
conda activate strangefish
export STOCKFISH_EXECUTABLE=/absolute/path/to/stockfish
export PYTHONPATH='.':$PYTHONPATH
```

## Local bot matches

See `scripts/example.py` to play a local RBC game using StrangeFish.

## Server matches

To connect to the server as was done for the NeurIPS 2019 tournament,
use `strangefish/utilities/modified_rc_connect.py`, implemented here with
click command-line arguments.

Ex: `python strangefish/utilities/modified_rc_connect.py [your username] [your password] --max-concurrent-games 1`
Or: `python strangefish/utilities/modified_rc_connect.py [your username] [your password] --ranked True`
