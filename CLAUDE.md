# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

**The Sentinel** is the wake word detection component of the larger ALAN (Autonomous Learning AI Nexus) system - a fully autonomous, locally-hosted AI assistant with a sarcastic personality inspired by Alan Turing. This component runs on distributed Raspberry Pi devices and communicates with the central orchestrator via Tailscale networking.

## Architecture Overview

### Component Role in ALAN System
- **Wake Word Sentinel**: Continuous listening for "Alan" and "Nexus" wake words
- **Distributed Edge Device**: Runs on Raspberry Pi with low-power monitoring
- **Network Integration**: Sends wake signals to central RTX 3060 orchestrator
- **Location Context**: Each sentinel identifies its physical room/location

### Key Design Patterns
- **Dual Operation**: Simultaneous wake word detection + FastAPI HTTP server
- **Network-First Architecture**: All communication over local network (Tailscale)
- **Resource Management**: Proper PyAudio/Porcupine cleanup and threading
- **Location-Aware Triggering**: Room context sent with every wake event

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set required environment variable
export PICOVOICE_ACCESS_KEY="your_key_here"
```

### Running the Service
```bash
# Run main sentinel service (blocks on wake word detection)
python sentinel.py
```

### Configuration
- Edit `config.yaml` for orchestrator endpoint, audio settings, location
- Place wake word models (.ppn files) in `../shared/wake-words/`
- Tailscale networking configured externally

## Core Architecture Components

### WakeWordSentinel Class
- **Porcupine Integration**: Custom .ppn models for "alan" and "nexus" 
- **PyAudio Streaming**: 16kHz, 16-bit PCM, frame-based processing
- **Network Communication**: POST requests to orchestrator `/wake` endpoint
- **Resource Cleanup**: Proper shutdown handling for audio streams

### SentinelAPI Class  
- **FastAPI Server**: Runs on port 8090 for audio recording and playback
- **Audio Recording**: On-demand WAV file generation via `/record` endpoint
- **Audio Playback**: Receive and play audio files via `/play` endpoint
- **File Serving**: Download recorded audio via `/audio/{session_id}`
- **Health Monitoring**: `/health` endpoint for system status

### Threading Model
```python
# Main thread: Wake word detection (blocking)
sentinel.listen()

# Background daemon thread: FastAPI server
server_thread = threading.Thread(target=api.start_server, daemon=True)
```

## Key Configuration Points

### Network Integration (config.yaml)
```yaml
orchestrator:
  host: "theserver.tail054cdb.ts.net"  # Tailscale MagicDNS
  port: 8080
  endpoint: "/wake"
```

### Location Context
```yaml
location:
  room: "kitchen"  # Unique identifier for this sentinel
  description: "Kitchen Pi Sentinel"
```

### Audio Configuration
```yaml
audio:
  sample_rate: 16000
  frame_length: 512
  device_index: null  # null = default device
```

## Integration with ALAN System

### Wake Event Payload
```python
payload = {
    'wake_word': 'nexus',           # Detected wake word
    'location': 'kitchen',          # Room identifier
    'timestamp': time.time(),       # Detection timestamp
    'sentinel_info': {
        'room': 'kitchen',
        'description': 'Kitchen Pi Sentinel'
    }
}
```

### Data Flow in Larger System
```
1. Sentinel detects "Alan" in kitchen → Network signal to Orchestrator
2. Orchestrator requests audio recording from kitchen sentinel
3. Audio processed by AI Brain (RTX 3060) → Response generated  
4. Voice service synthesizes TTS audio → POST to kitchen Pi /play endpoint
5. Sentinel plays audio through Pi speakers
```

## Dependencies and Requirements

### Hardware Requirements
- Raspberry Pi 4 (primary deployment target)
- USB microphone (tested: Anker PowerPro)
- Network connectivity to orchestrator system

### Critical Dependencies
- **Picovoice Porcupine**: Wake word detection (requires API key)
- **Custom .ppn Models**: Must exist in `../shared/wake-words/` directory
- **Tailscale**: Network connectivity (configured externally)

### Python Dependencies
- PyAudio for audio capture
- FastAPI + Uvicorn for HTTP API
- Requests for orchestrator communication
- PyYAML for configuration loading

## Development and Testing

### Validation Checklist
1. **Wake Word Detection**: Test actual "nexus"/"alan" utterances
2. **Network Communication**: Verify orchestrator connectivity via Tailscale
3. **Audio Recording**: Test `/record` endpoint functionality
4. **Resource Cleanup**: Ensure proper shutdown handling
5. **Concurrent Operations**: Verify wake detection during HTTP recording

### Common Issues
- **Audio Device Problems**: Check `device_index` in config, verify microphone permissions
- **Network Connectivity**: Confirm Tailscale status, orchestrator availability
- **Wake Word Issues**: Verify .ppn models exist, check PICOVOICE_ACCESS_KEY
- **Missing Models**: Startup fails if wake word .ppn files not found

## Error Handling Strategy

### Network Failures
- Log but don't crash on orchestrator communication failures
- Continue wake word detection despite network issues
- Graceful degradation when orchestrator unavailable

### Audio Failures  
- Fatal errors on audio device initialization (fail-fast)
- Proper cleanup of PyAudio resources on all exit paths
- Separate audio streams for wake detection vs recording to avoid conflicts

## Performance Characteristics

### Target Metrics (per ALAN system requirements)
- **Wake Word Latency**: < 500ms detection
- **Network Response**: Quick notification to orchestrator
- **Resource Usage**: Low-power operation suitable for Pi
- **Reliability**: Continuous operation with graceful error handling

### Audio Processing
- Real-time frame-based processing (512 samples per frame)
- No buffering delays in wake word detection path
- Separate recording streams to avoid interference