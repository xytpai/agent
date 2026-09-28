## AMD-GPU VLLM User Guide

#### 1. Start server

```bash
export MODEL_NAME=Qwen/Qwen2.5-32B-Instruct
export SERVE_PORT=8900
export SERVE_TP=1
```

```bash
vllm serve ${MODEL_NAME} \
  --host 0.0.0.0 \
  --port ${SERVE_PORT} \
  --max-model-len 32768 \
  --seed 0 \
  --tensor-parallel-size ${SERVE_TP} \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype auto \
  --no-enable-prefix-caching \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --performance-mode balanced \
  --optimization-level 2 \
  --max-logprobs 20 \
  --generation-config auto \
  --uvicorn-log-level info \
  --max-log-len 256
```
