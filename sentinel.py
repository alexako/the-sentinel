#!/usr/bin/env python3
import os
import pvporcupine
import pyaudio
import requests
import yaml
import logging
import time
import asyncio
import threading
import wave
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import aiofiles

class WakeWordSentinel:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.porcupine = None
        self.audio_stream = None
        self.pa = None
        self.running = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def initialize(self):
        """Initialize Porcupine and audio stream"""
        try:
            # Initialize Porcupine with wake words
            wake_words = self.config['sentinel']['wake_words']
            self.logger.info(f"Initializing with wake words: {wake_words}")
            
            # Load wake word models from shared directory
            model_paths = []
            shared_dir = Path("../shared/wake-words")
            
            for wake_word in wake_words:
                model_path = shared_dir / f"{wake_word}.ppn"
                if model_path.exists():
                    model_paths.append(str(model_path))
                else:
                    self.logger.error(f"Wake word model not found: {model_path}")
                    raise FileNotFoundError(f"Missing wake word model: {wake_word}.ppn")
            
            access_key = os.getenv('PICOVOICE_ACCESS_KEY')
            if not access_key:
                raise ValueError("PICOVOICE_ACCESS_KEY environment variable not set")
            
            self.porcupine = pvporcupine.create(
                keyword_paths=model_paths,
                access_key=access_key
            )
            
            # Initialize PyAudio
            self.pa = pyaudio.PyAudio()
            
            # Get audio device info
            device_index = self.config['sentinel']['audio']['device_index']
            if device_index is None:
                device_index = self.pa.get_default_input_device_info()['index']
            
            # Create audio stream
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.porcupine.frame_length
            )
            
            self.logger.info("Sentinel initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sentinel: {e}")
            raise
    
    def notify_orchestrator(self, wake_word: str) -> bool:
        """Send wake word detection to orchestrator"""
        try:
            orchestrator_config = self.config['sentinel']['orchestrator']
            location_config = self.config['sentinel']['location']
            
            url = f"http://{orchestrator_config['host']}:{orchestrator_config['port']}{orchestrator_config['endpoint']}"
            
            payload = {
                'wake_word': wake_word,
                'location': location_config['room'],
                'timestamp': time.time(),
                'sentinel_info': {
                    'room': location_config['room'],
                    'description': location_config['description']
                }
            }
            
            response = requests.post(
                url,
                json=payload,
                timeout=orchestrator_config['timeout']
            )
            
            if response.status_code == 200:
                self.logger.info(f"Successfully notified orchestrator: {wake_word} from {location_config['room']}")
                return True
            else:
                self.logger.warning(f"Orchestrator responded with status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to notify orchestrator: {e}")
            return False
    
    def listen(self):
        """Main listening loop"""
        if not self.porcupine or not self.audio_stream:
            raise RuntimeError("Sentinel not initialized. Call initialize() first.")
        
        self.running = True
        self.logger.info("Starting wake word detection...")
        
        try:
            while self.running:
                # Read audio frame
                pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm = [int.from_bytes(pcm[i:i+2], byteorder='little', signed=True) 
                       for i in range(0, len(pcm), 2)]
                
                # Process with Porcupine
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    wake_word = self.config['sentinel']['wake_words'][keyword_index]
                    self.logger.info(f"Wake word detected: {wake_word}")
                    
                    # Play immediate audio feedback for nexus wake word
                    if wake_word.lower() == 'nexus':
                        self._play_nexus_processing_sound()
                    
                    # Notify orchestrator
                    self.notify_orchestrator(wake_word)
                    
        except KeyboardInterrupt:
            self.logger.info("Stopping sentinel...")
        except Exception as e:
            self.logger.error(f"Error in listening loop: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Clean up resources"""
        self.running = False
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if self.pa:
            self.pa.terminate()
        
        if self.porcupine:
            self.porcupine.delete()
        
        self.logger.info("Sentinel stopped")
    
    def _play_nexus_processing_sound(self):
        """Play futuristic processing sound immediately upon nexus wake word detection"""
        try:
            self.logger.info("Playing nexus processing sound...")
            
            # Generate the same futuristic sound effect as in orchestrator
            import struct
            import math
            import tempfile
            
            sample_rate = 16000
            duration = 0.5  # Shorter duration for immediate feedback
            duration_samples = int(sample_rate * duration)
            
            # Create WAV header (44 bytes)
            wav_header = bytearray(44)
            
            # RIFF header
            wav_header[0:4] = b'RIFF'
            # File size (filled later)
            wav_header[8:12] = b'WAVE'
            
            # fmt chunk
            wav_header[12:16] = b'fmt '
            wav_header[16:20] = (16).to_bytes(4, 'little')  # chunk size
            wav_header[20:22] = (1).to_bytes(2, 'little')   # PCM format
            wav_header[22:24] = (1).to_bytes(2, 'little')   # mono
            wav_header[24:28] = sample_rate.to_bytes(4, 'little')
            wav_header[28:32] = (sample_rate * 2).to_bytes(4, 'little')  # byte rate
            wav_header[32:34] = (2).to_bytes(2, 'little')   # block align
            wav_header[34:36] = (16).to_bytes(2, 'little')  # bits per sample
            
            # data chunk header
            wav_header[36:40] = b'data'
            data_size = duration_samples * 2  # 16-bit samples
            file_size = len(wav_header) + data_size - 8
            
            # Update file size in header
            wav_header[4:8] = file_size.to_bytes(4, 'little')
            wav_header[40:44] = data_size.to_bytes(4, 'little')
            
            # Generate futuristic processing sound effect
            audio_data = bytearray()
            for i in range(duration_samples):
                time_position = i / sample_rate
                
                if time_position < 0.2:  # First 0.2s: rising sweep
                    frequency = 200 + (600 * time_position / 0.2)  # 200Hz to 800Hz sweep
                    amplitude = 1200 * (time_position / 0.2)  # Fade in (slightly louder for Pi speakers)
                elif time_position < 1.0:  # Next 0.8s: pulsing high-tech tone
                    frequency = 800 + 200 * math.sin(2 * math.pi * 3 * time_position)  # Modulated 800Hz
                    pulse = 0.7 + 0.3 * math.sin(2 * math.pi * 8 * time_position)  # Pulsing envelope
                    amplitude = 900 * pulse  # Louder for Pi speakers
                elif time_position < duration - 0.3:  # Middle section: subtle processing hum
                    frequency = 300 + 50 * math.sin(2 * math.pi * 0.5 * time_position)  # Slow modulation
                    amplitude = 400  # Quiet background hum
                else:  # Last 0.3s: fade out with descending tone
                    remaining = (duration - time_position) / 0.3
                    frequency = 400 * remaining  # Descending tone
                    amplitude = 500 * remaining  # Fade out
                
                # Generate the waveform (sounds like sci-fi computer processing)
                sample_value = int(amplitude * math.sin(2 * math.pi * frequency * time_position))
                
                # Clamp to 16-bit range
                sample_value = max(-32767, min(32767, sample_value))
                
                # Convert to 16-bit signed integer, little-endian
                audio_data.extend(struct.pack('<h', sample_value))
            
            # Write to temporary file and play
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(wav_header)
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Play the sound using system audio player (non-blocking)
            # Try multiple players in order of preference for Pi
            players = ['aplay', 'paplay', 'cvlc', 'mpg123']
            
            for player in players:
                try:
                    if player == 'aplay':
                        result = subprocess.run(['which', 'aplay'], capture_output=True)
                        if result.returncode == 0:
                            subprocess.Popen(['aplay', temp_file_path], 
                                           stdout=subprocess.DEVNULL, 
                                           stderr=subprocess.DEVNULL)
                            self.logger.info("Nexus processing sound played with aplay")
                            break
                    elif player == 'paplay':
                        result = subprocess.run(['which', 'paplay'], capture_output=True)
                        if result.returncode == 0:
                            subprocess.Popen(['paplay', temp_file_path], 
                                           stdout=subprocess.DEVNULL, 
                                           stderr=subprocess.DEVNULL)
                            self.logger.info("Nexus processing sound played with paplay")
                            break
                except Exception as e:
                    self.logger.debug(f"Failed to play with {player}: {e}")
                    continue
            
            # Clean up temp file after a delay (async cleanup)
            def cleanup_temp_file():
                import os
                time.sleep(duration + 1)  # Wait for playback to finish
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            threading.Thread(target=cleanup_temp_file, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Failed to play nexus processing sound: {e}")

class RecordingRequest(BaseModel):
    session_id: str
    duration: float = 5.0
    sample_rate: int = 16000

class RecordingResponse(BaseModel):
    session_id: str
    status: str
    message: str
    file_path: Optional[str] = None

class PlaybackRequest(BaseModel):
    session_id: str
    location: str
    volume: float = 0.7

class PlaybackResponse(BaseModel):
    session_id: str
    status: str
    message: str
    location: str

class SentinelAPI:
    def __init__(self, sentinel: WakeWordSentinel):
        self.sentinel = sentinel
        self.app = FastAPI(title="ALAN Sentinel Audio API")
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        self.playback_dir = Path("playback")
        self.playback_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup routes
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "service": "sentinel"}
        
        @self.app.get("/audio/{session_id}")
        async def get_audio(session_id: str):
            """Download recorded audio file"""
            audio_file = self.recordings_dir / f"{session_id}.wav"
            
            if not audio_file.exists():
                raise HTTPException(status_code=404, detail="Audio file not found")
            
            return FileResponse(
                path=str(audio_file),
                media_type="audio/wav",
                filename=f"{session_id}.wav"
            )
        
        @self.app.post("/record", response_model=RecordingResponse)
        async def record_audio(request: RecordingRequest):
            """Record audio for the specified duration"""
            try:
                self.logger.info(f"Recording request: {request.session_id}, {request.duration}s")
                
                # Record audio
                audio_file = await self.record_audio_clip(
                    request.session_id,
                    request.duration,
                    request.sample_rate
                )
                
                if audio_file:
                    return RecordingResponse(
                        session_id=request.session_id,
                        status="success",
                        message=f"Recorded {request.duration}s audio",
                        file_path=str(audio_file)
                    )
                else:
                    raise HTTPException(status_code=500, detail="Recording failed")
                    
            except Exception as e:
                self.logger.error(f"Recording failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/play", response_model=PlaybackResponse)
        async def play_audio(
            audio_file: UploadFile = File(...),
            session_id: str = None,
            location: str = "unknown",
            volume: float = 0.7
        ):
            """Receive and play audio file from Voice service"""
            try:
                # Generate session_id if not provided
                if not session_id:
                    session_id = f"play_{int(time.time())}"
                
                self.logger.info(f"Playback request: {session_id} for {location} (volume: {volume})")
                
                # Save uploaded audio file
                playback_file = self.playback_dir / f"{session_id}.wav"
                
                async with aiofiles.open(playback_file, 'wb') as f:
                    content = await audio_file.read()
                    await f.write(content)
                
                self.logger.info(f"Audio file saved: {playback_file} ({len(content)} bytes)")
                
                # Play the audio file
                success = await self.play_audio_file(playback_file, volume)
                
                if success:
                    return PlaybackResponse(
                        session_id=session_id,
                        status="success",
                        message=f"Audio played successfully at {location}",
                        location=location
                    )
                else:
                    raise HTTPException(status_code=500, detail="Audio playback failed")
                    
            except Exception as e:
                self.logger.error(f"Playback failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def record_audio_clip(self, session_id: str, duration: float, sample_rate: int) -> Optional[Path]:
        """Record audio clip using the same PyAudio setup as wake word detection"""
        try:
            # We need to temporarily pause wake word detection to record
            # For now, we'll create a separate audio stream for recording
            
            pa = pyaudio.PyAudio()
            
            # Create recording stream
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            self.logger.info(f"Recording {duration}s of audio...")
            
            # Record audio
            frames = []
            chunk_size = 1024
            sample_count = int(sample_rate * duration)
            chunks_needed = sample_count // chunk_size
            
            for _ in range(chunks_needed):
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            pa.terminate()
            
            # Save to WAV file
            audio_file = self.recordings_dir / f"{session_id}.wav"
            
            with wave.open(str(audio_file), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                wf.setframerate(sample_rate)
                wf.writeframes(b''.join(frames))
            
            self.logger.info(f"Audio recorded: {audio_file} ({audio_file.stat().st_size} bytes)")
            return audio_file
            
        except Exception as e:
            self.logger.error(f"Audio recording failed: {e}")
            return None
    
    async def play_audio_file(self, audio_file: Path, volume: float = 0.7) -> bool:
        """Play audio file using system audio player"""
        try:
            if not audio_file.exists():
                self.logger.error(f"Audio file not found: {audio_file}")
                return False
            
            # Try multiple audio players in order of preference
            players = ['aplay', 'paplay', 'cvlc', 'mpg123', 'ffplay']
            
            for player in players:
                if await self._check_command_exists(player):
                    self.logger.info(f"Playing audio with {player}: {audio_file}")
                    
                    # Build command based on player
                    if player == 'aplay':
                        cmd = ['aplay', str(audio_file)]
                    elif player == 'paplay':
                        cmd = ['paplay', '--volume', str(int(volume * 65536)), str(audio_file)]
                    elif player == 'cvlc':
                        cmd = ['cvlc', '--play-and-exit', '--intf', 'dummy', str(audio_file)]
                    elif player == 'mpg123':
                        cmd = ['mpg123', '-q', str(audio_file)]
                    elif player == 'ffplay':
                        cmd = ['ffplay', '-nodisp', '-autoexit', '-v', 'quiet', str(audio_file)]
                    else:
                        continue
                    
                    # Execute playback command
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        self.logger.info(f"Audio playback successful with {player}")
                        return True
                    else:
                        self.logger.warning(f"Audio playback failed with {player}: {stderr.decode()}")
                        continue
            
            self.logger.error("No working audio player found")
            return False
            
        except Exception as e:
            self.logger.error(f"Audio playback error: {e}")
            return False
    
    async def _check_command_exists(self, command: str) -> bool:
        """Check if a command exists on the system"""
        try:
            process = await asyncio.create_subprocess_exec(
                'which', command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.communicate()
            return process.returncode == 0
        except Exception:
            return False
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8090):
        """Start the FastAPI server"""
        self.logger.info(f"Starting Sentinel API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")

def main():
    sentinel = WakeWordSentinel()
    
    try:
        sentinel.initialize()
        
        # Create API server
        api = SentinelAPI(sentinel)
        
        # Start API server in a separate thread
        server_thread = threading.Thread(
            target=api.start_server,
            kwargs={"host": "0.0.0.0", "port": 8090},
            daemon=True
        )
        server_thread.start()
        
        # Give server time to start
        time.sleep(2)
        
        # Start wake word detection (this will block)
        sentinel.listen()
        
    except Exception as e:
        logging.error(f"Sentinel failed: {e}")
    finally:
        sentinel.stop()

if __name__ == "__main__":
    main()
