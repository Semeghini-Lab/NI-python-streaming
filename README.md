# NI Streamer

A Python package for streaming data to National Instruments DAQ cards, specifically designed for atom array control.

## Installation

For development (recommended):
```bash
pip install -e .
```

For production:
```bash
pip install .
```

## Usage

```python
from nistreamer import NICard, SequenceStreamer, AOSequence, DOSequence

# Create sequences
ch0 = AOSequence(channel_id="ao0", sample_rate=300_000)
ch0.linramp(0.0, 1.0, start=0, end=2)

# Create and configure cards
card = NICard(
    device_name="PXI1Slot3",
    sample_rate=300_000,
    sequences=[ch0],
)

# Compile the card
card.compile(chunk_size=65536)

# Stream the sequence
with SequenceStreamer(
    cards=[card],
    num_workers=4,
    num_writers=1,
    pool_size=8,
) as streamer:
    streamer.start()
```

## Development

The package is structured as follows:
- `nistreamer/`: Main package directory
  - `SequenceStreamer.py`: Main streaming manager
  - `NICard.py`: NI card interface
  - `Worker.py`: Worker process for computation
  - `Writer.py`: Writer process for DAQ output
  - `Sequences.py`: Sequence definitions
  - Other supporting modules

## Issues
- Multiple writers: attempting to use multiple errors will result in a PLL error. Currently, the user is forced to use a single writer.
    - On startup of the Writer process: the buffer for each card is initially filled, then the task associated with each card started. When initializing the buffer for each card, the card attempts to start the sample card. However, since the task for the primary card has not been started, it does not export its clock, and we get a PLL error. For unknown reasons, this is not an issue when a single process handle all of this, but becomes an issue when two processes handle this.

## Requirements
- Python >= 3.7
- numpy
- pyzmq
- nidaqmx (for NI-DAQ functionality) 