#!/bin/sh
# NightRun bootstrap installer.
#
#   curl -fsSL https://nightrun.io/install-nightrun.sh | sh
#
# This script does exactly two things, and nothing else:
#   1. git-clones https://github.com/hardrave/NIGHTRUN (or updates an
#      existing ./NIGHTRUN checkout),
#   2. starts the repository's interactive installer, ./install.sh.
#
# Everything that matters — model download and verification, image
# building, device selection, and the exact typed confirmation before
# any disk is written — happens in the interactive installer, which you
# can read here first:
#   https://github.com/hardrave/NIGHTRUN/blob/master/install.sh
#
# Prefer not to pipe to a shell? The equivalent is:
#   git clone https://github.com/hardrave/NIGHTRUN.git
#   cd NIGHTRUN && ./install.sh

set -u

REPO="https://github.com/hardrave/NIGHTRUN.git"
DIR="NIGHTRUN"

say() { printf '%s\n' "$*"; }

case "$(uname -s)" in
    Linux) ;;
    *) say "NightRun's installer supports Linux hosts only."; exit 1 ;;
esac

if ! command -v git >/dev/null 2>&1; then
    say "git is required. Install it (e.g. 'sudo apt install git') and re-run."
    exit 1
fi

if [ -d "$DIR/.git" ]; then
    say "Updating existing $DIR checkout..."
    git -C "$DIR" pull --ff-only || {
        say "Could not fast-forward $DIR; resolve manually and re-run."
        exit 1
    }
else
    say "Cloning $REPO ..."
    git clone "$REPO" "$DIR" || exit 1
fi

cd "$DIR" || exit 1

# When piped (curl | sh), stdin is the pipe — reattach the terminal so
# the interactive installer can actually ask questions.
if [ -t 0 ]; then
    exec ./install.sh
elif [ -r /dev/tty ]; then
    exec ./install.sh < /dev/tty
else
    say "No interactive terminal available."
    say "Run it directly instead:  cd $DIR && ./install.sh"
    exit 1
fi
