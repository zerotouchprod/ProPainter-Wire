import sys
import traceback
import runpod
from batch_inference import run_job

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

def handler(job):
    runpod_id = job["id"]
    inp = job["input"]
    source_url = inp.get("sourceUrl")

    print(f"Received Job ID: {runpod_id}", flush=True)

    if not source_url:
        return {"status": 0, "error": "Missing sourceUrl"}

    def update_progress(message):
        try:
            runpod.serverless.progress_update(job, message)
        except Exception:
            pass

    try:
        # returns dict: {'url': '...', 'duration': 123, 'errors': []}
        job_result = run_job(
            job_id=runpod_id, 
            source_url=source_url, 
            progress_callback=update_progress
        )

        # FIX: Unpack the dictionary keys explicitly for the UI
        return {
            "status": 1,
            "jobID": runpod_id,
            "resultUrl": job_result.get("url"),      # Map 'url' to 'resultUrl'
            "duration": job_result.get("duration"),  # Pass duration to top level
            "errors": job_result.get("errors", [])   # Pass errors to top level
        }

    except Exception as e:
        print(f"❌ Job {runpod_id} Failed!", flush=True)
        traceback.print_exc()
        return {
            "status": 0,
            "error": str(e)
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
