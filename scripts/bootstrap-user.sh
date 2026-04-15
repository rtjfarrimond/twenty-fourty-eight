#!/usr/bin/env bash
set -euo pipefail

# One-time bootstrap: create the 2048-solver system user/group and chown
# the data directories. Idempotent — safe to re-run.
#
# After running this:
#   1. Log out and back in (so your shell picks up the new group membership)
#   2. Re-run scripts/deploy.sh
#
# Usage: sudo ./scripts/bootstrap-user.sh [admin-username]
#   admin-username defaults to the invoking user (SUDO_USER).

SERVICE_USER="2048-solver"
DATA_DIR="/var/lib/2048-solver"
ADMIN_USER="${1:-${SUDO_USER:-}}"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: must be run as root (use sudo)" >&2
    exit 1
fi

if [ -z "$ADMIN_USER" ]; then
    echo "ERROR: could not determine admin username. Pass it as the first arg." >&2
    exit 1
fi

if ! id "$ADMIN_USER" >/dev/null 2>&1; then
    echo "ERROR: admin user '$ADMIN_USER' does not exist" >&2
    exit 1
fi

echo "=== Bootstrapping $SERVICE_USER ==="

# Group first (some distros' useradd --system creates a matching group, others don't)
if ! getent group "$SERVICE_USER" >/dev/null; then
    echo "--- Creating group $SERVICE_USER ---"
    groupadd --system "$SERVICE_USER"
else
    echo "--- Group $SERVICE_USER already exists ---"
fi

# System user (no shell, no home — daemon-only)
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    echo "--- Creating user $SERVICE_USER ---"
    useradd --system --no-create-home --shell /usr/sbin/nologin \
        --gid "$SERVICE_USER" "$SERVICE_USER"
else
    echo "--- User $SERVICE_USER already exists ---"
fi

# Add admin user to the group so they can write training data without sudo
if id -nG "$ADMIN_USER" | tr ' ' '\n' | grep -qx "$SERVICE_USER"; then
    echo "--- $ADMIN_USER already in $SERVICE_USER group ---"
else
    echo "--- Adding $ADMIN_USER to $SERVICE_USER group ---"
    usermod -a -G "$SERVICE_USER" "$ADMIN_USER"
    echo "    NOTE: log out and back in for the new group to take effect"
fi

# Data dir: owned by the service user, group-writable, setgid so new files
# inherit the group (otherwise files admin creates would be owned by their
# own group and the service couldn't read them).
echo "--- Setting up $DATA_DIR ---"
mkdir -p "$DATA_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR"
chmod -R g+w "$DATA_DIR"
find "$DATA_DIR" -type d -exec chmod g+s {} \;

echo
echo "=== Bootstrap complete ==="
echo "Next steps:"
echo "  1. Log out and back in so '$ADMIN_USER' picks up the '$SERVICE_USER' group"
echo "  2. Run scripts/deploy.sh to install the systemd unit with the new user"
