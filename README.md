# Deep Hawkes Processes for High-Impact Event Alpha

This project explores a Neural Hawkes Process for modeling high-frequency limit order book events from LOBSTER data and using the learned event intensities as a signal for simple alpha generation.

The codebase includes:

- LOBSTER message parsing into event sequences
- A continuous-time LSTM cell for asynchronous event modeling
- Neural Hawkes training with negative log-likelihood loss
- Intensity visualization for buy/sell executions
- A simple event-driven backtest that combines Hawkes intensities with order book imbalance

## Repository Structure

```text
.
├── Dataset/
│   ├── AMZN_2012-06-21_34200000_57600000_message_5.csv
│   ├── AMZN_2012-06-21_34200000_57600000_orderbook_5.csv
│   └── LOBSTER_SampleFiles_ReadMe.txt
├── backtest.py
├── cell.py
├── data.py
├── evaluate.py
├── hawkes_weights.pth
├── loss.py
├── model.py
└── train.py
```

## What The Project Does

The model treats order book activity as a sequence of timestamped discrete events. Each event is embedded, passed through a continuous-time LSTM cell, and converted into event intensities. Those intensities are then used to:

- fit the model via Hawkes-style log-likelihood
- visualize predicted buy/sell pressure over time
- drive a simple trading rule in the backtest

## Data

The included sample data is LOBSTER-format message and order book data for AMZN:

- `Dataset/AMZN_2012-06-21_34200000_57600000_message_5.csv`
- `Dataset/AMZN_2012-06-21_34200000_57600000_orderbook_5.csv`

Expected message columns:

1. `Time`
2. `Type`
3. `OrderID`
4. `Size`
5. `Price`
6. `Direction`

The dataset loader computes `TimeDelta` from event timestamps and maps raw message types into model event classes used by the Neural Hawkes model.

## Model Overview

### 1. `cell.py`

Implements `CTLSTMCell`, a continuous-time LSTM cell that:

- decays the cell state as a function of elapsed time
- updates hidden state when a new event arrives
- supports irregularly spaced event streams

### 2. `model.py`

Defines `NeuralHawkesModel`, which contains:

- an event embedding layer
- the continuous-time LSTM cell
- a linear intensity head with `softplus` activation

The forward pass returns:

- hidden states
- decay rates
- base event intensities

### 3. `loss.py`

Defines `HawkesLogLikelihoodLoss`, which approximates the negative log-likelihood by combining:

- reward for assigning high intensity to the observed event
- penalty for maintaining high total intensity between events

## Training

Train the model with:

```bash
python3 train.py
```

Default training configuration in `train.py`:

- `NUM_EVENT_TYPES = 5`
- `BATCH_SIZE = 64`
- `SEQ_LEN = 50`
- `EMBEDDING_DIM = 16`
- `HIDDEN_DIM = 64`
- `EPOCHS = 30`
- `LEARNING_RATE = 1e-3`

The script automatically selects:

- `mps` on Apple Silicon if available
- `cuda` if available
- otherwise `cpu`

At the end of training, weights are saved to:

```text
hawkes_weights.pth
```

## Evaluation And Visualization

To visualize price, true execution events, and predicted buy/sell intensities:

```bash
python3 evaluate.py
```

`evaluate.py`:

- loads the first 1000 events from the LOBSTER message file
- runs sequential inference using the trained model
- plots:
  - price
  - true market buy/sell executions
  - predicted buy and sell intensities

## Backtesting

Run the simple strategy backtest with:

```bash
python3 backtest.py
```

The backtest:

- streams message and top-of-book data
- computes order book imbalance (OBI)
- estimates buy and sell execution intensity from the Hawkes model
- enters long when:
  - predicted buy intensity is high
  - OBI is sufficiently positive
  - current inventory is flat
- exits when:
  - predicted sell intensity is high
  - OBI is sufficiently negative

The script prints rolling mark-to-market updates and a final summary including:

- total trades
- fees paid
- ending capital
- net profit

## Dependencies

This repository does not currently include a `requirements.txt`, but the code imports:

- `torch`
- `pandas`
- `numpy`
- `matplotlib`

A typical setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch pandas numpy matplotlib
```

## Typical Workflow

1. Place LOBSTER message and order book CSVs in `Dataset/`
2. Train the model with `python3 train.py`
3. Inspect learned intensities with `python3 evaluate.py`
4. Run the toy strategy with `python3 backtest.py`

## Known Caveats

This repo is a solid experimental prototype, but there are a few things worth knowing before relying on the outputs:
