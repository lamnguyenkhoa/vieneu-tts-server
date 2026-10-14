@echo off
rem Starts the VieNeu TTS server from the folder this file is in.
rem To start it when you log in, put a shortcut to this file (not a copy) in the
rem folder that "shell:startup" opens (Win+R, shell:startup).
rem The server runs in its own console window; close that window to stop it.

start "VieNeu TTS Server" /min /d "%~dp0." cmd /k uv run python -m app.main
