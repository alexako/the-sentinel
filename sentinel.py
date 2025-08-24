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
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

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

class RecordingRequest(BaseModel):
    session_id: str
    duration: float = 5.0
    sample_rate: int = 16000

class RecordingResponse(BaseModel):
    session_id: str
    status: str
    message: str
    file_path: Optional[str] = None

class SentinelAPI:
    def __init__(self, sentinel: WakeWordSentinel):
        self.sentinel = sentinel
        self.app = FastAPI(title="ALAN Sentinel Audio API")
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        
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