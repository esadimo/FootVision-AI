#!/bin/zsh
# Double-click in Finder to start the FootVision web app; it opens in your browser.
# Close this window (or press Ctrl+C) to stop it.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/streamlit ]; then
  echo "The Python environment (.venv) with streamlit is missing — see README 'Setup'."
  read -k1 "?Press any key to close."
  exit 1
fi
exec .venv/bin/streamlit run app.py --server.headless false --browser.gatherUsageStats false
