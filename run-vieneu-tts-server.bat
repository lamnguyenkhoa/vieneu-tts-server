@echo off
setlocal
rem Double-click to install dependencies (first run) and start the server.
rem To start it when you log in, put a shortcut to this file (not a copy) in the
rem folder that "shell:startup" opens (Win+R, shell:startup). Set the shortcut's
rem Run option to Minimized to keep the window out of the way.
title VieNeu TTS Server
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    ) else (
        echo Installing uv...
        powershell -NoProfile -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    )
)

echo Installing dependencies...
uv sync --frozen
if errorlevel 1 goto :error

rem Settings (HOST, PORT, TTS_API_KEY, ...) come from .env if present; see .env.example.
rem The server reads .env itself; variables already set in the environment take precedence.
echo Starting server...
uv run --frozen python -m app.main
if errorlevel 1 goto :error
goto :eof

:error
echo.
echo Something went wrong. See the messages above.
pause
exit /b 1
