# Known Bugs

### Deploy blocked by running training
Deploy script can't overwrite `/opt/2048-solver/bin/training` while a training
process is running ("Text file busy"). Deploy stops the server but not training.
