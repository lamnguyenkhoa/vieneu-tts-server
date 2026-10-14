@echo off
rem Starts the VieNeu TTS server when you log in.
rem Copy this file to the folder that "shell:startup" opens (Win+R, shell:startup).
rem The server runs in a minimized console window; close that window to stop it.

start "VieNeu TTS Server" /min /d "D:\python-projects\vieneu-tts-server" cmd /k uv run python -m app.main
