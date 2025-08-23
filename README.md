# The Sentinel

Wake word detection service for ALAN system. Runs on Raspberry Pi, listens for "Alan" and "Nexus" wake words, and notifies the orchestrator via network.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Get Picovoice access key from https://picovoice.ai/
3. Update config.yaml with your orchestrator IP and access key
4. Place wake word models in ../shared/wake_words/
5. Run: `python sentinel.py`

## Configuration

Edit `config.yaml`:
- Set orchestrator host/port
- Configure audio device
- Set location identifier (room name)
- Add Picovoice access key

## Hardware

- Raspberry Pi 4
- USB microphone (Anker PowerPro or similar)
- Network connection to orchestrator