# Future Ideas

### Proper user/group for services and data
Currently `/var/lib/2048-solver/` is root-owned and the server runs as root.
Rob has been chowned the data dirs as a workaround to avoid sudo for training.
Proper fix: create a `2048-solver` system user + group, run the server as
that user, add rob to the group, chmod g+w the data dirs. Update the systemd
unit and deploy script accordingly.

### Training job queue
Persistent queue of training runs that survives reboots, lets you inspect
what's pending, reorder, and kill jobs. Today we chain with `&&`, which is
fine for sequential overnight runs but can't be introspected or edited.
