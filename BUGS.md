# Known Bugs

### Deploy blocked by running training
Deploy script can't overwrite `/opt/2048-solver/bin/training` while a training
process is running ("Text file busy"). Deploy stops the server but not training.

### Charts don't restore size after exiting fullscreen
Exiting fullscreen leaves the chart container at full page width instead of
returning to the grid layout. Fixed on refresh.

### Live training dashboard only shows up to 4096 rate
Stats bar shows 2048 and 4096 tile rates but not 8192/16384/32768. Should match
the results table which goes up to 32768.

### Chart x-axis wrong for mixed eval intervals
On the results page, models with different eval intervals (10K vs 100K) share
x-axis labels from the longest curve. Shorter models get plotted by array index,
not by actual games_trained value, so the scale is wrong.
