import os
import whisper
import tempfile
import subprocess
import wave
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import List, Tuple, Dict, Any
import time
import json
import torch
import numpy as np

# Open source speaker diarization using SpeechBrain, VAD, and spectral clustering
_speechbrain_available = False
_sklearn_available = False
_librosa_available = False

def _init_open_source_diarization():
    """Initialize open source diarization dependencies"""
    global _speechbrain_available, _sklearn_available, _librosa_available
    
    # Check SpeechBrain availability
    try:
        from speechbrain.inference.speaker import SpeakerRecognition
        from speechbrain.inference.VAD import VAD
        _speechbrain_available = True
    except Exception:
        _speechbrain_available = False
    
    # Check sklearn availability
    try:
        from sklearn.cluster import SpectralClustering, AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity
        _sklearn_available = True
    except Exception:
        _sklearn_available = False
    
    # Check librosa availability
    try:
        import librosa
        _librosa_available = True
    except Exception:
        _librosa_available = False
    
    return _speechbrain_available or _sklearn_available

# Global model variables for multiprocessing
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


def try_speechbrain_diarization(audio_path: str) -> Dict[str, Any]:
    """Try SpeechBrain open source diarization"""
    try:
        from speechbrain.inference.speaker import SpeakerRecognition
        from speechbrain.inference.VAD import VAD
        
        print("🔄 Using SpeechBrain Speaker Diarization...")
        
        # Load VAD model
        vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmp_vad")
        
        # Load speaker verification model
        verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="tmp_spkrec"
        )
        
        # Process audio
        boundaries = vad.get_speech_segments(audio_path)
        
        if len(boundaries) < 2:
            return {"speakers": {}, "segments": [], "error": "Not enough speech segments"}
        
        embeddings = []
        segments_info = []
        
        for i, (start, end) in enumerate(boundaries):
            try:
                embedding = verification.encode_batch_from_file(
                    audio_path, 
                    start_sample=int(start*16000), 
                    end_sample=int(end*16000)
                )
                embeddings.append(embedding.squeeze().cpu().numpy())
                segments_info.append({"start": start, "end": end, "duration": end-start})
            except Exception as e:
                print(f"Warning: Failed to process segment {i+1}: {e}")
                continue
        
        if len(embeddings) >= 2:
            return cluster_speakers_with_embeddings(embeddings, segments_info)
        
        return {"speakers": {}, "segments": [], "error": "Insufficient embeddings"}
        
    except Exception as e:
        return {"speakers": {}, "segments": [], "error": f"SpeechBrain failed: {e}"}

def try_spectral_clustering_diarization(audio_path: str) -> Dict[str, Any]:
    """Try spectral clustering approach using audio features"""
    try:
        import librosa
        from sklearn.cluster import SpectralClustering
        from sklearn.preprocessing import StandardScaler
        
        print("🔄 Using Spectral Clustering Diarization...")
        
        # Load and preprocess audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Extract features in 2-second windows
        hop_length = sr * 2
        n_segments = len(y) // hop_length
        
        if n_segments < 2:
            return {"speakers": {}, "segments": [], "error": "Audio too short for clustering"}
        
        features = []
        segments_info = []
        
        for i in range(n_segments):
            start_sample = i * hop_length
            end_sample = min((i + 1) * hop_length, len(y))
            segment = y[start_sample:end_sample]
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Extract other features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(segment))
            
            # Combine features
            feature_vector = np.concatenate([mfcc_mean, [spectral_centroid, zero_crossing_rate]])
            features.append(feature_vector)
            
            segments_info.append({
                "start": start_sample / sr,
                "end": end_sample / sr,
                "duration": (end_sample - start_sample) / sr
            })
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Cluster into speakers (max 2 for sales calls)
        n_speakers = min(2, len(features))
        clustering = SpectralClustering(n_clusters=n_speakers, random_state=42)
        speaker_labels = clustering.fit_predict(features_normalized)
        
        # Create speaker results
        speakers = {}
        segments = []
        
        for i, (segment, label) in enumerate(zip(segments_info, speaker_labels)):
            speaker_id = f"SPEAKER_{label:02d}"
            
            # Add speaker info to segment
            segment_with_speaker = segment.copy()
            segment_with_speaker["speaker"] = speaker_id
            segments.append(segment_with_speaker)
            
            # Track speaker stats
            if speaker_id not in speakers:
                speakers[speaker_id] = {
                    "total_duration": 0,
                    "segments_count": 0
                }
            
            speakers[speaker_id]["total_duration"] += segment["duration"]
            speakers[speaker_id]["segments_count"] += 1
        
        return {
            "speakers": speakers,
            "segments": segments,
            "total_speakers": len(speakers)
        }
        
    except Exception as e:
        return {"speakers": {}, "segments": [], "error": f"Spectral clustering failed: {e}"}

def try_simple_vad_diarization(audio_path: str) -> Dict[str, Any]:
    """Try simple VAD + alternating speaker assignment"""
    try:
        print("🔄 Using Simple VAD Diarization...")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Simple energy-based VAD
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        # Calculate energy for each frame
        frames = []
        for i in range(0, waveform.shape[1] - frame_length, hop_length):
            frame = waveform[0, i:i+frame_length]
            energy = torch.sum(frame ** 2)
            frames.append(energy.item())
        
        frames = np.array(frames)
        
        # Simple VAD threshold
        threshold = np.mean(frames) + 0.5 * np.std(frames)
        speech_frames = frames > threshold
        
        # Convert frame indices to time segments
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                start_time = start_frame * hop_length / sample_rate
                end_time = i * hop_length / sample_rate
                if end_time - start_time > 0.5:  # Minimum 0.5s segments
                    segments.append({
                        "start": start_time,
                        "end": end_time,
                        "duration": end_time - start_time
                    })
                in_speech = False
        
        if len(segments) < 2:
            return {"speakers": {}, "segments": [], "error": "Not enough speech segments"}
        
        # Simple alternating speaker assignment
        speakers = {
            "SPEAKER_00": {"total_duration": 0, "segments_count": 0},
            "SPEAKER_01": {"total_duration": 0, "segments_count": 0}
        }
        
        diarized_segments = []
        for i, segment in enumerate(segments):
            speaker_id = f"SPEAKER_{i % 2:02d}"
            segment_with_speaker = segment.copy()
            segment_with_speaker["speaker"] = speaker_id
            diarized_segments.append(segment_with_speaker)
            
            speakers[speaker_id]["total_duration"] += segment["duration"]
            speakers[speaker_id]["segments_count"] += 1
        
        return {
            "speakers": speakers,
            "segments": diarized_segments,
            "total_speakers": len(speakers)
        }
        
    except Exception as e:
        return {"speakers": {}, "segments": [], "error": f"Simple VAD failed: {e}"}

def cluster_speakers_with_embeddings(embeddings, segments_info):
    """Cluster speakers using embeddings and cosine similarity"""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.cluster import AgglomerativeClustering
        
        # Calculate similarity matrix
        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Use agglomerative clustering
        n_speakers = min(2, len(embeddings))
        clustering = AgglomerativeClustering(n_clusters=n_speakers, metric='precomputed', linkage='average')
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        speaker_labels = clustering.fit_predict(distance_matrix)
        
        # Organize results
        speakers = {}
        segments = []
        
        for i, (segment, label) in enumerate(zip(segments_info, speaker_labels)):
            speaker_id = f"SPEAKER_{label:02d}"
            
            segment_with_speaker = segment.copy()
            segment_with_speaker["speaker"] = speaker_id
            segments.append(segment_with_speaker)
            
            if speaker_id not in speakers:
                speakers[speaker_id] = {
                    "total_duration": 0,
                    "segments_count": 0
                }
            
            speakers[speaker_id]["total_duration"] += segment["duration"]
            speakers[speaker_id]["segments_count"] += 1
        
        return {
            "speakers": speakers,
            "segments": segments,
            "total_speakers": len(speakers)
        }
        
    except Exception as e:
        return {"speakers": {}, "segments": [], "error": f"Clustering failed: {e}"}

def perform_open_source_diarization(audio_path: str, timeout_seconds: int = 60) -> Dict[str, Any]:
    """
    Perform open source speaker diarization using multiple fallback approaches
    
    Args:
        audio_path: Path to audio file
        timeout_seconds: Maximum time to wait for diarization (unused for now)
        
    Returns:
        Dictionary with speaker segments and timeline
    """
    print("🎭 Performing open source speaker diarization...")
    
    # Try different approaches in order of preference
    approaches = [
        (try_speechbrain_diarization, "SpeechBrain"),
        (try_spectral_clustering_diarization, "Spectral Clustering"),
        (try_simple_vad_diarization, "Simple VAD")
    ]
    
    for approach_func, approach_name in approaches:
        try:
            result = approach_func(audio_path)
            if not result.get("error") and result.get("speakers"):
                print(f"✅ {approach_name} diarization succeeded")
                print(f"🎯 Found {len(result['speakers'])} unique speakers")
                return result
            else:
                print(f"⚠️  {approach_name} failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"❌ {approach_name} exception: {e}")
    
    print("❌ All diarization approaches failed")
    return {"speakers": {}, "segments": [], "error": "All diarization methods failed"}


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
def transcribe_audio(file_path: str, language: str = "hi", use_parallel: bool = True, chunk_duration: int = 30, include_diarization: bool = True) -> Tuple[str, List, Dict]:
    """
    Main transcription function with option for parallel processing and speaker diarization
    
    Args:
        file_path: Path to input audio file
        language: Language code for transcription
        use_parallel: Whether to use parallel processing
        chunk_duration: Duration of each chunk in seconds (for parallel mode)
        include_diarization: Whether to perform speaker diarization
    
    Returns:
        Tuple of (full_text, segments, diarization_info)
    """
    # Perform transcription
    if use_parallel:
        # Get audio duration to decide if parallel processing is worth it
        duration = get_audio_duration_ffmpeg(file_path)
        
        # Only use parallel processing for longer audio files
        if duration > 60:  # Only parallelize if audio is longer than 1 minute
            full_text, segments = transcribe_audio_parallel(file_path, language, chunk_duration)
        else:
            full_text, segments = transcribe_audio_single(file_path, language)
    else:
        # Fall back to original single-threaded approach
        full_text, segments = transcribe_audio_single(file_path, language)
    
    # Perform speaker diarization if requested
    diarization_info = {}
    if include_diarization:
        # Convert to WAV for diarization if needed
        wav_path = convert_to_wav(file_path) if not file_path.lower().endswith('.wav') else file_path
        
        try:
            diarization_info = perform_open_source_diarization(wav_path)
            
            # Create speaker-labeled transcription
            if diarization_info.get('segments') and not diarization_info.get('error'):
                print("🔗 Creating speaker-labeled transcription...")
                
                # For now, create a simple timeline-based speaker assignment
                # This works well for sales calls with alternating speakers
                diar_segments = sorted(diarization_info['segments'], key=lambda x: x['start'])
                
                if len(diar_segments) > 0:
                    # Method 1: Timeline-based assignment (simpler and more reliable)
                    speaker_timeline = []
                    for diar_seg in diar_segments:
                        speaker_timeline.append({
                            'start': diar_seg['start'],
                            'end': diar_seg['end'], 
                            'speaker': diar_seg['speaker']
                        })
                    
                    # Assign speakers to transcription segments based on timeline
                    for trans_seg in segments:
                        assigned_speaker = "UNKNOWN"
                        trans_mid = (trans_seg['start'] + trans_seg['end']) / 2
                        
                        # Find which speaker segment this transcription falls into
                        for timeline_seg in speaker_timeline:
                            if timeline_seg['start'] <= trans_mid <= timeline_seg['end']:
                                assigned_speaker = timeline_seg['speaker']
                                break
                        
                        trans_seg['speaker'] = assigned_speaker
                    
                    # Create speaker-labeled full text
                    speaker_labeled_parts = []
                    current_speaker = None
                    current_speaker_text = []
                    
                    for segment in segments:
                        speaker = segment.get('speaker', 'UNKNOWN')
                        text = segment.get('text', '').strip()
                        
                        if text:
                            if speaker != current_speaker:
                                # New speaker, save previous and start new
                                if current_speaker_text and current_speaker:
                                    combined_text = ' '.join(current_speaker_text)
                                    speaker_labeled_parts.append(f"[{current_speaker}]: {combined_text}")
                                
                                current_speaker = speaker
                                current_speaker_text = [text]
                            else:
                                # Same speaker, accumulate text
                                current_speaker_text.append(text)
                    
                    # Don't forget the last speaker
                    if current_speaker_text and current_speaker:
                        combined_text = ' '.join(current_speaker_text)
                        speaker_labeled_parts.append(f"[{current_speaker}]: {combined_text}")
                    
                    if speaker_labeled_parts:
                        full_text = "\n\n".join(speaker_labeled_parts)
                        print(f"✅ Created speaker-labeled transcription with {len(speaker_labeled_parts)} speaker turns")
        
        except Exception as e:
            print(f"Warning: Diarization failed: {e}")
            diarization_info = {"error": str(e)}
        
        finally:
            # Cleanup WAV file if we created it
            if wav_path != file_path:
                try:
                    os.remove(wav_path)
                except:
                    pass
    
    return full_text, segments, diarization_info

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file_path> [--parallel] [--chunk-duration=30] [--no-diarization]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    use_parallel = "--parallel" in sys.argv or True  # Default to parallel
    include_diarization = "--no-diarization" not in sys.argv  # Default to include diarization
    
    # Parse chunk duration
    chunk_duration = 30
    for arg in sys.argv:
        if arg.startswith("--chunk-duration="):
            chunk_duration = int(arg.split("=")[1])
    
    print("🔄 Transcribing audio with speaker diarization...")
    try:
        start_time = time.time()
        full_text, segments, diarization_info = transcribe_audio(
            audio_file, 
            language="hi", 
            use_parallel=use_parallel, 
            chunk_duration=chunk_duration,
            include_diarization=include_diarization
        )
        end_time = time.time()
        
        print(f"\n✅ Transcription Complete in {end_time - start_time:.2f} seconds:\n")
        print(full_text)
        
        # Print diarization summary
        if include_diarization and diarization_info.get('speakers'):
            print(f"\n🎭 Speaker Summary:")
            for speaker, info in diarization_info['speakers'].items():
                print(f"  {speaker}: {info['total_duration']:.1f}s ({info['segments_count']} segments)")
        
        # Save to txt file
        out_path = audio_file + ".txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"\n📝 Transcription saved to: {out_path}")
        
        # Save detailed JSON output if diarization was performed
        if include_diarization:
            json_out_path = audio_file + "_detailed.json"
            detailed_output = {
                "full_text": full_text,
                "segments": segments,
                "diarization_info": diarization_info,
                "processing_time": end_time - start_time
            }
            with open(json_out_path, "w", encoding="utf-8") as f:
                json.dump(detailed_output, f, indent=2, ensure_ascii=False)
            print(f"📊 Detailed output saved to: {json_out_path}")
        
    except Exception as e:
        print(f"❌ Error during transcription:\n{e}")


