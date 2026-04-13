# Known Bugs

### Deploy blocked by running training
Deploy script can't overwrite `/opt/2048-solver/bin/training` while a training
process is running ("Text file busy"). Deploy stops the server but not training.

### Charts don't restore size after exiting fullscreen
Exiting fullscreen leaves the chart container at full page width instead of
returning to the grid layout. Fixed on refresh.

### Chart x-axis wrong for mixed eval intervals
On the results page, models with different eval intervals (10K vs 100K) share
x-axis labels from the longest curve. Shorter models get plotted by array index,
not by actual games_trained value, so the scale is wrong.
