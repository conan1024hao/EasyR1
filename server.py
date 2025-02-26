from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# Load the model and tokenizer
MODEL_NAME = "ModelSpace/GemmaX2-28-2B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()

language_id2name = {
    "zh": "Chinese",
}

class RequestData(BaseModel):
    text: str
    max_length: int = 512
    language_code: str = "zh"

@app.post("/generate")
async def generate_text(request_data: RequestData):
    prompt = (f"Translate this from English to {language_id2name[request_data.language_code]}:\n"
              f"English: {request_data.text}\n{language_id2name[request_data.language_code]}:")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=request_data.max_length)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"response": output_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1314)