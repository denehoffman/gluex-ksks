default:
  just --list

venv:
  [ -d .venv ] || uv venv --python=3.13
  source .venv/bin/activate

setup: venv
  uv pip install --reinstall -e .

get-data queue:
  ./.venv/bin/run-dselectors --queue {{queue}}
