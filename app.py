from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from transcribe import transcribe_audio  # Your new optimized transcribe function
from llm_analysis import analyze_transcription_with_ollama
import time

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    start_time = time.time()
    
    file = request.files["file"]
    if not file:
        return "No file uploaded", 400
    
    # Save uploaded file
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename.replace(" ", "_")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    print(f"📁 File saved: {file_path}")
    
    try:
        # Use optimized parallel transcription
        transcription_start = time.time()
        full_text, segments = transcribe_audio(
            file_path, 
            language="hi", 
            use_parallel=True,  # Enable parallel processing
            chunk_duration=30   # 30-second chunks
        )
        transcription_time = time.time() - transcription_start
        
        print(f"🎯 Transcription completed in {transcription_time:.2f} seconds")
        print(f"📝 Transcribed text: {full_text[:200]}...")  # First 200 chars
        
        # Get LLM analysis
        analysis_start = time.time()
        llm_output = analyze_transcription_with_ollama(full_text)
        analysis_time = time.time() - analysis_start
        
        print(f"🤖 LLM analysis completed in {analysis_time:.2f} seconds")
        
        total_time = time.time() - start_time
        
        # Log performance metrics
        print(f"⏱️  Performance Summary:")
        print(f"   - Transcription: {transcription_time:.2f}s")
        print(f"   - LLM Analysis: {analysis_time:.2f}s")
        print(f"   - Total Time: {total_time:.2f}s")
        
        # Cleanup uploaded file (optional)
        try:
            os.remove(file_path)
        except:
            pass
        
        return render_template("index.html", 
                             llm_output=llm_output, 
                             full_text=full_text,
                             processing_time=f"{total_time:.2f}",
                             transcription_time=f"{transcription_time:.2f}",
                             analysis_time=f"{analysis_time:.2f}")
    
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return f"Error processing file: {str(e)}", 500

@app.route("/upload_async", methods=["POST"])
def upload_file_async():
    """
    Alternative async endpoint that returns JSON for AJAX calls
    This allows for better user experience with progress updates
    """
    start_time = time.time()
    
    file = request.files["file"]
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename.replace(" ", "_")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    try:
        # Transcription
        transcription_start = time.time()
        full_text, segments = transcribe_audio(
            file_path, 
            language="hi", 
            use_parallel=True,
            chunk_duration=30
        )
        transcription_time = time.time() - transcription_start
        
        # LLM Analysis
        analysis_start = time.time()
        llm_output = analyze_transcription_with_ollama(full_text)
        analysis_time = time.time() - analysis_start
        
        total_time = time.time() - start_time
        
        # Cleanup
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify({
            "success": True,
            "full_text": full_text,
            "llm_output": llm_output,
            "performance": {
                "transcription_time": round(transcription_time, 2),
                "analysis_time": round(analysis_time, 2),
                "total_time": round(total_time, 2)
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, threaded=True)  # Enable threading for Flask