@echo off

start cmd /k "uv run python -m main"

start cmd /k "uv run streamlit run web/app.py"