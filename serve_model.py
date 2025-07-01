import argparse, asyncio, time
from typing import List, Any, Dict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = FastAPI()


class Texts(BaseModel):
    texts: List[str]


class BatchedPredictor:
    def __init__(self, hf_pipeline, max_batch_size: int = 32, max_latency_ms: int = 10):
        self._pipeline = hf_pipeline
        self._max_batch_size = max_batch_size
        self._max_latency_ms = max_latency_ms / 1000.0
        self._queue: List = []
        self._lock = asyncio.Lock()
        self._bg_task = asyncio.create_task(self._batch_worker())

    async def predict(self, text: str):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        async with self._lock:
            self._queue.append((text, fut))
            if len(self._queue) >= self._max_batch_size:
                await self._flush()
        return await fut

    async def _batch_worker(self):
        while True:
            await asyncio.sleep(self._max_latency_ms)
            async with self._lock:
                if self._queue:
                    await self._flush()

    async def _flush(self):
        texts, futures = zip(*self._queue)
        self._queue = []
        outputs = self._pipeline(list(texts))
        for out, fut in zip(outputs, futures):
            fut.set_result(out)


@app.on_event("startup")
async def startup_event():
    global predictor
    model_name = args.model
    device_arg = args.device
    if device_arg == "auto":
        if torch.cuda.is_available():
            device_arg = "cuda"
        elif torch.backends.mps.is_available():
            device_arg = "mps"
        else:
            device_arg = "cpu"

    if device_arg == "cuda":
        dtype = torch.float16 if args.half_precision else torch.float32
        torch.set_default_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if device_arg == "cuda":
        model.to("cuda")
    elif device_arg == "mps":
        model.to("mps")
    if args.half_precision and device_arg in {"cuda", "mps"}:
        model.half()
    if args.torch_compile:
        model = torch.compile(model)

    hf_pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if device_arg == "cuda" else -1,
        batch_size=args.max_batch_size,
        top_k=2,
    )
    predictor = BatchedPredictor(hf_pipe, max_batch_size=args.max_batch_size, max_latency_ms=args.latency_ms)


@app.post("/predict")
async def predict(texts: Texts):
    try:
        results = await asyncio.gather(*[predictor.predict(t) for t in texts.texts])
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server():
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve HF encoder model with efficient batching")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--max_batch_size", type=int, default=32)
    parser.add_argument("--latency_ms", type=int, default=10, help="Max batch latency in ms")
    parser.add_argument("--half_precision", action="store_true", help="Use float16 where possible")
    parser.add_argument("--torch_compile", action="store_true", help="Compile the model with torch.compile")

    args = parser.parse_args()

    run_server()