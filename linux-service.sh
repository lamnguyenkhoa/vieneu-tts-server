sudo tee /etc/systemd/system/vieneu-tts.service > /dev/null <<'EOF'
[Unit]
Description=VieNeu TTS Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=atoma
WorkingDirectory=/home/atoma/Atoma/vieneu-tts-server
ExecStart=/home/atoma/.local/bin/uv run python -m app.main
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF