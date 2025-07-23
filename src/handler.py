import os
import time
import runpod
from vllm import LLM, SamplingParams
from huggingface_hub import hf_hub_download
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm = None
model_name = None

def download_gguf_model():
    """Download GGUF model to local storage if needed"""
    model_name = os.getenv("MODEL_NAME", "mistralai/Devstral-Small-2507_gguf")
    gguf_filename = os.getenv("GGUF_FILENAME", "Devstral-Small-2507-Q5_K_M.gguf")
    
    # Check if it's a local path already
    if os.path.exists(model_name):
        logger.info(f"Using local model path: {model_name}")
        return model_name
    
    # Check if it's a GGUF repo
    if "GGUF" in model_name or gguf_filename:
        logger.info(f"Downloading GGUF model: {model_name}")
        model_path = "/tmp/model"
        os.makedirs(model_path, exist_ok=True)
        
        try:
            local_file = hf_hub_download(
                repo_id=model_name,
                filename=gguf_filename,
                local_dir=model_path,
                token=os.getenv("HUGGING_FACE_HUB_TOKEN")
            )
            full_path = os.path.join(model_path, gguf_filename)
            logger.info(f"Downloaded GGUF model to: {full_path}")
            return full_path
        except Exception as e:
            logger.error(f"Failed to download GGUF model: {e}")
            raise
    
    # For non-GGUF models, return as-is
    return model_name

def initialize_model():
    """Initialize the vLLM model"""
    global llm, model_name
    
    if llm is not None:
        return
    
    logger.info("Initializing vLLM model...")
    model_path = download_gguf_model()
    
    # vLLM initialization parameters
    vllm_kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "max_model_len": int(os.getenv("MAX_MODEL_LEN", "4096")),
        "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
    }
    
    # Add tensor parallel size if specified
    tensor_parallel_size = os.getenv("TENSOR_PARALLEL_SIZE")
    if tensor_parallel_size:
        vllm_kwargs["tensor_parallel_size"] = int(tensor_parallel_size)
    
    # Add quantization if specified
    quantization = os.getenv("QUANTIZATION")
    if quantization:
        vllm_kwargs["quantization"] = quantization
    
    try:
        llm = LLM(**vllm_kwargs)
        model_name = model_path
        logger.info(f"Successfully initialized model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

def handler(job):
    """
    Handler function for RunPod serverless.
    """
    global llm
    
    try:
        # Initialize model if not already done
        if llm is None:
            initialize_model()
        
        # Extract job input
        job_input = job["input"]
        
        # Extract parameters
        prompt = job_input.get("prompt", "")
        max_tokens = job_input.get("max_tokens", 100)
        temperature = job_input.get("temperature", 0.7)
        top_p = job_input.get("top_p", 1.0)
        top_k = job_input.get("top_k", -1)
        stop = job_input.get("stop", None)
        
        if not prompt:
            return {"error": "No prompt provided"}
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop
        )
        
        # Generate response
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        generation_time = time.time() - start_time
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text
        
        return {
            "output": generated_text,
            "generation_time": generation_time,
            "model": model_name
        }
        
    except Exception as e:
        logger.error(f"Error in handler: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # For local testing 
    test_input = {
        "input": {
            "prompt": "def fibonacci(n):",
            "max_tokens": 100,
            "temperature": 0.1
        }
    }
    result = handler(test_input)
    print(result)

# Start the serverless function
runpod.serverless.start({"handler": handler})
