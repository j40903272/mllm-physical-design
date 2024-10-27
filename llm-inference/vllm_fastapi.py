import argparse
import json
import ssl
import uuid
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.pooling_params import PoolingParams
#from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 300  # seconds.
app = FastAPI()


engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.put("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()

    # trtllm to vllm
    if 'tokens_to_generate' in request_dict:
        request_dict['max_tokens'] = request_dict.pop("tokens_to_generate")
    if 'random_seed' in request_dict:
        request_dict.pop("random_seed")
    #    request_dict['seed'] = request_dict.pop("random_seed")
    if 'stop_words_list' in request_dict:
        request_dict['stop'] = request_dict.pop("stop_words_list")

    prompts = request_dict.pop("prompts")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    assert engine is not None

    if isinstance(prompts, str):

        results_generator = engine.generate(prompts, sampling_params, request_id)

        # Streaming case
        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for request_output in results_generator:
                prompt = request_output.prompt
                text_outputs = [
                    prompt[idx] + output.text for idx, output in enumerate(request_output.outputs)
                ]
                ret = {"text": text_outputs}
                yield (json.dumps(ret) + "\0").encode("utf-8")

        if stream:
            return StreamingResponse(stream_results())

        # Non-streaming case
        final_output = None
        async for request_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)
                return Response(status_code=499)
            print(request_output)
            final_output = request_output

        assert final_output is not None
        prompt = final_output.prompt
        text_outputs = [prompt[idx] + output.text for idx, output in enumerate(final_output.outputs)]
        return JSONResponse(text_outputs)
    
    elif isinstance(prompts, list):

        import asyncio

        async def run(results_generator):
            final_output = None
            async for request_output in results_generator:
                if await request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await engine.abort(request_id)
                    return Response(status_code=499)
                final_output = request_output

            assert final_output is not None
            prompt = final_output.prompt
            return [prompt[idx] + output.text for idx, output in enumerate(final_output.outputs)][0][1:]
            
        results = await asyncio.gather(*[
            run(engine.generate(prompt, sampling_params, f"{request_id}_{e}"))
            for e, prompt in enumerate(prompts)
        ])
        return results


@app.put("/encode")
async def encode(request: Request) -> Response:
    """Encode the input text and return embeddings.

    The request should be a JSON object with the following fields:
    - input: the text to encode.
    """
    request_dict = await request.json()

    # Extract input text from request
    input_text = request_dict.get("input", "")

    # Generate a unique request_id using uuid
    request_id = str(uuid.uuid4())

    # Start the encoding process
    results_generator = engine.encode(input_text, PoolingParams(), request_id)

    final_output = None

    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None

    # Assuming the final_output contains the embeddings as `embeddings`
    embeddings = final_output.embeddings

    # Return the embeddings as a JSON response
    return JSONResponse(content={"embeddings": embeddings})


def setup_vllm_engine(model: str, tokenizer: str):
    global engine
    args = AsyncEngineArgs(model)
    args.tokenizer = tokenizer
    args.dtype = 'bfloat16'
    args.max_model_len = 4096
    args.device="auto"
    args.gpu_memory_utilization = 0.9
    # args.pipeline_parallel_size=4
    args.worker_use_ray=True
    args.tensor_parallel_size = 8
    args.disable_log_requests=True
    args.disable_log_stats=True
    engine = AsyncLLMEngine.from_engine_args(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-6.7b-instruct")
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/deepseek-coder-6.7b-instruct")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")
    # parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    
    setup_vllm_engine(args.model, args.tokenizer)
    app.root_path = args.root_path
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        # ssl_keyfile=args.ssl_keyfile,
        # ssl_certfile=args.ssl_certfile,
        # ssl_ca_certs=args.ssl_ca_certs,
        # ssl_cert_reqs=args.ssl_cert_reqs
    )
