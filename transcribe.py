import os
import whisper
import tempfile
import subprocess
import wave
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import math
from typing import List, Tuple
import time
import json

# Global model variable for multiprocessing
model = None

def init_worker():
    """Initialize Whisper model in each worker process"""
    global model
    model = whisper.load_model("large")

def get_audio_duration_ffmpeg(file_path: str) -> float:
    """Get audio duration in seconds using ffmpeg"""
    try:
        command = [
            "ffprobe", "-v", "quiet", "-print_format", "json", 
            "-show_format", file_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        return duration
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0

def convert_to_wav(file_path: str, output_path: str = None) -> str:
    """Convert audio file to WAV format using ffmpeg"""
    if output_path is None:
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        output_path = temp_wav.name
        temp_wav.close()

    command = [
        "ffmpeg", "-y", "-i", file_path,
        "-ar", "16000", "-ac", "1", "-f", "wav",
        output_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    
    return output_path

def create_audio_chunks_ffmpeg(file_path: str, chunk_duration: int = 30) -> List[str]:
    """
    Split audio file into chunks using ffmpeg
    Returns list of chunk file paths
    """
    # Get total duration
    total_duration = get_audio_duration_ffmpeg(file_path)
    
    chunk_paths = []
    current_time = 0
    chunk_index = 0
    
    while current_time < total_duration:
        # Create temp file for this chunk
        chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk_{chunk_index}.wav")
        chunk_path = chunk_file.name
        chunk_file.close()
        
        # Extract chunk using ffmpeg
        command = [
            "ffmpeg", "-y",
            "-i", file_path,
            "-ss", str(current_time),  # Start time
            "-t", str(chunk_duration),  # Duration
            "-ar", "16000", "-ac", "1", "-f", "wav",
            chunk_path
        ]
        
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            # Check if chunk has actual audio content
            if os.path.getsize(chunk_path) > 1024:  # More than 1KB
                chunk_paths.append(chunk_path)
            else:
                os.remove(chunk_path)  # Remove empty chunk
        else:
            print(f"Warning: Failed to create chunk {chunk_index}")
            try:
                os.remove(chunk_path)
            except:
                pass
        
        current_time += chunk_duration
        chunk_index += 1
    
    return chunk_paths

def transcribe_chunk(chunk_info: Tuple[str, int, str]) -> Tuple[int, str, List]:
    """
    Transcribe a single audio chunk
    Returns (chunk_index, transcribed_text, segments)
    """
    chunk_path, chunk_index, language = chunk_info
    global model
    
    try:
        result = model.transcribe(
            chunk_path,
            language=language,
            verbose=False,
            condition_on_previous_text=False,
            temperature=0.5
        )
        
        return chunk_index, result["text"], result["segments"]
    
    except Exception as e:
        print(f"Error transcribing chunk {chunk_index}: {e}")
        return chunk_index, "", []

def merge_transcriptions(chunk_results: List[Tuple[int, str, List]], chunk_duration: int = 30) -> Tuple[str, List]:
    """
    Merge transcription results from multiple chunks
    Returns (full_text, all_segments)
    """
    # Sort by chunk index
    chunk_results.sort(key=lambda x: x[0])
    
    full_text_parts = []
    all_segments = []
    
    for chunk_index, text, segments in chunk_results:
        if text.strip():
            full_text_parts.append(text.strip())
        
        # Adjust segment timestamps and add to all_segments
        time_offset = chunk_index * chunk_duration
        for segment in segments:
            adjusted_segment = segment.copy()
            adjusted_segment['start'] += time_offset
            adjusted_segment['end'] += time_offset
            all_segments.append(adjusted_segment)
    
    full_text = " ".join(full_text_parts)
    return full_text, all_segments

def transcribe_audio_parallel(file_path: str, language: str = "hi", chunk_duration: int = 30, max_workers: int = None) -> Tuple[str, List]:
    """
    Main function to transcribe audio using parallel processing
    
    Args:
        file_path: Path to input audio file
        language: Language code for transcription
        chunk_duration: Duration of each chunk in seconds
        max_workers: Maximum number of parallel processes (None = CPU count)
    
    Returns:
        Tuple of (full_text, segments)
    """
    start_time = time.time()
    
    print(f"📁 Input file: {file_path}")
    
    # Get audio duration
    duration = get_audio_duration_ffmpeg(file_path)
    print(f"⏱️  Audio duration: {duration:.2f} seconds")
    
    # Convert to WAV first
    wav_path = convert_to_wav(file_path)
    print(f"🎧 Converted to WAV: {wav_path}")
    
    # Create chunks using ffmpeg
    print(f"✂️  Creating {chunk_duration}-second chunks...")
    chunk_paths = create_audio_chunks_ffmpeg(wav_path, chunk_duration)
    print(f"📦 Created {len(chunk_paths)} chunks")
    
    if not chunk_paths:
        raise RuntimeError("No valid audio chunks created")
    
    # Prepare chunk info for parallel processing
    chunk_info_list = [(chunk_path, i, language) for i, chunk_path in enumerate(chunk_paths)]
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(chunk_paths))
    
    print(f"🚀 Starting parallel transcription with {max_workers} workers...")
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        chunk_results = list(executor.map(transcribe_chunk, chunk_info_list))
    
    # Merge results
    print("🔄 Merging transcription results...")
    full_text, all_segments = merge_transcriptions(chunk_results, chunk_duration)
    
    # Cleanup temporary files
    for chunk_path in chunk_paths:
        try:
            os.remove(chunk_path)
        except:
            pass
    
    try:
        os.remove(wav_path)
    except:
        pass
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"✅ Transcription complete in {processing_time:.2f} seconds")
    print(f"⚡ Speed improvement: ~{len(chunk_paths)}x theoretical maximum")
    
    return full_text, all_segments

def is_valid_audio(file_path):
    """
    Checks whether the given .wav file has non-zero duration.
    """
    try:
        with wave.open(file_path, "rb") as wav_file:
            return wav_file.getnframes() > 0
    except:
        return False

# Original single-threaded function (kept for compatibility)
def transcribe_audio_single(file_path: str, language: str = "hi") -> Tuple[str, List]:
    """Original single-threaded transcription function"""
    print(f"📁 Input file: {file_path}")
    wav_path = convert_to_wav(file_path)
    print(f"🎧 Converted to WAV: {wav_path}")

    if not is_valid_audio(wav_path):
        os.remove(wav_path)
        raise RuntimeError("❌ Converted audio is empty or corrupt.")

    # Load model if not already loaded
    global model
    if model is None:
        model = whisper.load_model("large")

    result = model.transcribe(
        wav_path,
        language=language,
        verbose=False,
        condition_on_previous_text=False,
        temperature=0.5
    )

    os.remove(wav_path)  # Cleanup
    return result["text"], result["segments"]

# Updated main transcribe function (backward compatible)
def transcribe_audio(file_path: str, language: str = "hi", use_parallel: bool = True, chunk_duration: int = 30) -> Tuple[str, List]:
    """
    Main transcription function with option for parallel processing
    
    Args:
        file_path: Path to input audio file
        language: Language code for transcription
        use_parallel: Whether to use parallel processing
        chunk_duration: Duration of each chunk in seconds (for parallel mode)
    
    Returns:
        Tuple of (full_text, segments)
    """
    if use_parallel:
        # Get audio duration to decide if parallel processing is worth it
        duration = get_audio_duration_ffmpeg(file_path)
        
        # Only use parallel processing for longer audio files
        if duration > 60:  # Only parallelize if audio is longer than 1 minute
            return transcribe_audio_parallel(file_path, language, chunk_duration)
    
    # Fall back to original single-threaded approach for short files
    return transcribe_audio_single(file_path, language)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file_path> [--parallel] [--chunk-duration=30]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    use_parallel = "--parallel" in sys.argv or True  # Default to parallel
    
    # Parse chunk duration
    chunk_duration = 30
    for arg in sys.argv:
        if arg.startswith("--chunk-duration="):
            chunk_duration = int(arg.split("=")[1])
    
    print("🔄 Transcribing audio...")
    try:
        start_time = time.time()
        full_text, segments = transcribe_audio(audio_file, language="hi", use_parallel=use_parallel, chunk_duration=chunk_duration)
        end_time = time.time()
        
        print(f"\n✅ Transcription Complete in {end_time - start_time:.2f} seconds:\n")
        print(full_text)
        
        # Save to txt file
        out_path = audio_file + ".txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"\n📝 Transcription saved to: {out_path}")
        
    except Exception as e:
        print(f"❌ Error during transcription:\n{e}")




# You are a **Conversation Analysis AI** trained specifically for the **real estate industry**. Your task is to analyze a real estate sales call transcript to assess agent performance, identify customer interest, and provide insights for improvement.

# ### Instructions:

# 1. **Conversation Summary**
#    Provide a clear summary in **3 to 5 bullet points**, covering:

#    * Main topics discussed
#    * Customer's sentiment and level of interest
#    * Any objections or buying signals raised

# 2. **Agent Performance Evaluation**
#    Rate the agent on a **scale of 1 to 5** (1 = poor, 5 = excellent) across the following areas. Include a **brief justification (1–2 sentences)** for each score:

#    * `introduction_rapport`: First impression, greeting, rapport building
#    * `product_knowledge`: Accuracy and detail in describing the property or offering
#    * `objection_handling`: How effectively concerns or questions were addressed
#    * `tone_language`: Clarity, tone, professionalism, and active listening
#    * `closure_strategy`: Ending the call with a strong CTA, follow-up plan, or next steps

# 3. **Customer Buying Intent**
#    Classify the customer’s intent as one of the following:

#    * `"Not Interested"`
#    * `"Mildly Interested"`
#    * `"Interested but Hesitant"`
#    * `"Ready to Proceed"`

#    Provide a 1–2 sentence justification based on language and tone.

# 4. **Actionable Recommendations**
#    Suggest **2 to 4 specific, actionable improvements** the agent could apply in future calls to improve engagement, close more deals, or handle concerns better.

# 5. **Keyword & Phrase Spotting**
#    Extract and categorize important keywords or phrases from the transcript under:

#    * `"positive_signals"`: Indicates buying interest or intent
#    * `"objections_or_concerns"`: Hesitations, doubts, or questions
#    * `"sales_opportunities"`: Timing, needs, location, budget cues
#    * `"red_flags"`: Signs of disinterest or deal blockers

# ---

# ### 📦 Return the output in this **structured JSON** format:

# ```json
# {
#   "summary": [
#     "Bullet point 1",
#     "Bullet point 2",
#     "..."
#   ],
#   "agent_evaluation": {
#     "introduction_rapport": {
#       "rating": 1-5,
#       "justification": "..."
#     },
#     "product_knowledge": {
#       "rating": 1-5,
#       "justification": "..."
#     },
#     "objection_handling": {
#       "rating": 1-5,
#       "justification": "..."
#     },
#     "tone_language": {
#       "rating": 1-5,
#       "justification": "..."
#     },
#     "closure_strategy": {
#       "rating": 1-5,
#       "justification": "..."
#     }
#   },
#   "customer_intent": {
#     "classification": "Not Interested / Mildly Interested / Interested but Hesitant / Ready to Proceed",
#     "justification": "..."
#   },
#   "recommendations": [
#     "Recommendation 1",
#     "Recommendation 2",
#     "... up to 4"
#   ],
#   "keywords": {
#     "positive_signals": ["..."],
#     "objections_or_concerns": ["..."],
#     "sales_opportunities": ["..."],
#     "red_flags": ["..."]
#   }
# }
# ```

# ---

# **Transcript:**

# ```plaintext
# "{transcript}"
# ```