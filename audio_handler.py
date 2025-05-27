import os 
import time 
import threading # 'threading' for managing concurrent audio recording.
import numpy as np 
import pyaudio # 'pyaudio' for interacting with audio hardware.
import wave # 'wave' for reading/writing WAV files.
import sounddevice as sd # 'sounddevice' for playing audio.
import soundfile as sf # 'soundfile' for reading/writing audio files.
import queue # 'queue' for thread-safe audio buffering.
import logging 
import re

logger = logging.getLogger(__name__) 

class AudioHandler:
    """
    Manages audio input (recording) and output (Text-to-Speech playback)
    for the Voice RAG Agent.
    """
    def __init__(self, config, asr_model, tts_model, handle_user_query_callback):

        # Initializes the AudioHandler with configuration, ASR model, TTS model, and user query handler.
        self.config = config
        self.asr_model = asr_model
        self.tts = tts_model
        self.handle_user_query = handle_user_query_callback

        # Audio recording variables
        self.audio_queue = queue.Queue() # Initializes a thread-safe queue for buffering audio data during recording.
        self.is_recording = False 
        self.recorded_frames = [] # List: Stores raw audio frames recorded from the microphone.
        self.silence_threshold = 0.01 
        self.silence_counter = 0  # Counter: Tracks consecutive silence duration during recording to determine when to stop recording.

    def start_listening(self, agent_state):
        """
        Starts listening for audio input from the microphone.
        """
        if self.is_recording:
            logger.warning("Already recording audio.")
            return "I'm already listening."

        agent_state.set_active(True)
        agent_state.mark_waiting_for_input(True) 

        logger.info("Starting audio recording...") 
        self.is_recording = True  # Sets the internal flag to indicate that recording has begun.
        self.recorded_frames = []  # Clears any previously recorded audio frames to start new conversation.
        self.silence_counter = 0

        threading.Thread(target=self._record_audio, daemon=True).start() # Starts a new thread to handle audio recording without blocking the main thread.
        return f"{self.config['agent_name']} is listening. Please speak your query."

    def stop_listening(self, agent_state):
        """
        Stops listening to voice input.
        """
        if not self.is_recording: 
            logger.warning("Not currently recording audio.") 
            return "I'm not listening right now." 

        logger.info("Stopping audio recording...")
        self.is_recording = False
        agent_state.mark_waiting_for_input(False) 
        return "Stopped listening. Processing your request..."

    def _record_audio(self):
        """
        Records audio from the microphone with improved adaptive silence detection.
        This method runs in a separate thread.
        """
        chunk_size = 1024
        format = pyaudio.paInt16  
        channels = 1 
        rate = self.config["audio_sample_rate"] 

        audio = pyaudio.PyAudio()

        stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)

        logger.info("Recording started. You can start speaking.")

        silence_counter = 0
        had_speech = False  # Flag to indicate if any speech has been detected since recording started.
        max_silence_duration = self.config.get("audio_input_timeout", 5.0) 
        min_speech_length = 1.0  # Minimum duration of speech required to consider the input valid (in seconds).
        speech_duration = 0

        # Ambient Noise Sampling 
        ambient_frames = []
        for _ in range(10):
            data = stream.read(chunk_size)
            ambient_frames.append(np.frombuffer(data, dtype=np.int16)) # Converts raw bytes to numpy array and appends.

        # Calculates the average absolute amplitude of ambient noise, normalized to a float between 0 and 1.
        ambient_level = np.mean([np.abs(frame).mean() for frame in ambient_frames]) / 32768.0 # Average ambient noise volume.
        # Sets the adaptive silence threshold: it's either the initial self.silence_threshold or 1.5 times the ambient noise level, whichever is greater.
        silence_threshold = max(self.silence_threshold, ambient_level * 1.5)
        
        logger.info(f"Adaptive silence threshold set to: {silence_threshold:.4f}") # Logs the calculated adaptive silence threshold.
        
        while self.is_recording: 
            data = stream.read(chunk_size)
            self.recorded_frames.append(data)

            # Check for silence with the adaptive threshold.
            audio_data = np.frombuffer(data, dtype=np.int16) 
            volume_norm = np.abs(audio_data).mean() / 32768.0 
            is_speech = volume_norm > silence_threshold

            if is_speech:
                had_speech = True
                silence_counter = 0
                speech_duration += chunk_size / rate # Adds the duration of the current chunk to total speech duration.
            else: # If silence is detected.
                silence_counter += chunk_size / rate # Increments silence counter by the duration of the current chunk.
                
                if had_speech and speech_duration > min_speech_length and silence_counter >= max_silence_duration:
                    logger.info(f"Silence detected after {speech_duration:.2f}s of speech. Stopping recording.") 
                    self.is_recording = False # To stop the recording loop.

        stream.stop_stream()
        stream.close()
        audio.terminate()

        if self.recorded_frames: 
            # Save recorded audio to a temporary file.
            temp_filename = os.path.join(self.config["tts_temp_dir"], "temp_recording.wav") 
            wf = wave.open(temp_filename, 'wb') 
            wf.setnchannels(channels) 
            wf.setsampwidth(audio.get_sample_size(format)) 
            wf.setframerate(rate) 
            wf.writeframes(b''.join(self.recorded_frames)) # Writes all recorded audio frames to the WAV file.
            wf.close() 

    
            # The process_audio method will run in a separate thread to not block the main execution.
            threading.Thread(target=self.process_audio, args=(temp_filename,), daemon=True).start()
        else: # If no audio frames were recorded.
            logger.warning("No audio recorded.") 
            
    def process_audio(self, audio_file):
        """
        Processes the recorded audio file by transcribing it using ASR
        and then passing the transcription to the main agent for handling.
        """
        logger.info("Processing recorded audio with ASR...") 
        start_time = time.time()

        # Uses beam search for better transcription quality.
        segments, info = self.asr_model.transcribe(audio_file, beam_size=5, language="en", task="transcribe")

        transcription = "".join([segment.text for segment in segments]) # Joins all transcribed segments into a single string.

        asr_time = time.time() - start_time
        logger.info(f"ASR processing time: {asr_time:.2f} seconds. Transcription: {transcription}")

        self.handle_user_query(transcription)

    def text_to_speech(self, text):
        """
        Converts text to speech and plays it. Includes cleaning for LLM meta-instructions
        and sentence splitting for better TTS quality.
        """
        logger.info("Converting text to speech...") 
        output_file = os.path.join(self.config["tts_temp_dir"], "response.wav") 
        
        # Additional check to filter out meta-instructions that might have leaked through LLM output.
        patterns_to_remove = [ # List of regex patterns for cleaning/removing text before TTS.
            r"Try to answer.*detail\.", 
            r"This helps the user.*\.", 
            r"User:.*\?", 
            r"Respond with information about.*\.",
            r"(User|Assistant):.*" 
        ]
        
        cleaned_text = text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE|re.DOTALL) # Applies each pattern to the text.
        
        if not cleaned_text.strip(): # Checks if the cleaned text is empty or just whitespace. Ensuring that we haven't removed all content.
            cleaned_text = "I'm sorry, I wasn't able to generate a proper response. Could you please ask your question again?"
        
        # Break text into sentences to avoid TTS issues with very long, continuous text.
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        combined_audio = None
        sample_rate = self.config["audio_sample_rate"]

        for sentence in sentences: 
            if not sentence.strip(): 
                continue
            
            # TTS generation for each sentence.
            try:
                self.tts.tts_to_file(text=sentence, file_path=output_file) # Generates speech for the current sentence and saves it to a temporary file.
                
                if combined_audio is None: 
                    combined_audio, sample_rate = sf.read(output_file)
                else:
                    sentence_audio, _ = sf.read(output_file) # Reads the audio data for the current sentence.
                    combined_audio = np.append(combined_audio, sentence_audio) # Appends the current sentence's audio to the combined audio array.
                    
            except Exception as e: 
                logger.error(f"TTS error with sentence '{sentence}': {str(e)}") 
        
        if combined_audio is not None: 
            sf.write(output_file, combined_audio, sample_rate)
            
            # Play the audio.
            try:
                logger.info("Playing audio response...")
                sd.play(combined_audio, sample_rate)
                sd.wait()
            except Exception as e:
                logger.error(f"Error playing audio: {str(e)}")
        else:
            logger.error("Failed to generate speech")
