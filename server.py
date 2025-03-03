from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

# Initialize FastAPI app and LLM
app = FastAPI()
llm = LLM("ModelSpace/GemmaX2-28-2B-v0.1", tensor_parallel_size=8, gpu_memory_utilization=0.1)

# Constants
LANGUAGE_MAP = {
    "zh": "Chinese",
    "he": "Hebrew",
}

class RequestData(BaseModel):
    text: str
    max_length: int = 512
    language_code: str = "zh"

@app.post("/generate")
async def generate_text(request_data: RequestData):
    # Create sampling params from the request
    params = SamplingParams(max_tokens=request_data.max_length)
    
    # Build translation prompt
    target_language = LANGUAGE_MAP.get(request_data.language_code, "Chinese")
    prompt = f"Translate this from English to {target_language}:\nEnglish: {request_data.text}\n{target_language}:"
    
    # Generate and extract text
    outputs = llm.generate(prompt, sampling_params=params)
    output_text = outputs[0].outputs[0].text
    print(output_text)
    return {"response": output_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1314)