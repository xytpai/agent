## AToy

An agent create a lot shit

### Backend Test

```
export BASE_URL=${YOUR_LLM_GATEWAY_URL}
export API_KEY=${YOUR_LLM_GATEWAY_API_KEY}
export MODEL_NAME=${YOUR_LLM_MODEL_NAME}

python3 core/backends.py "Introduce yourself"
```

### Run agent

```bash
python3 core/agent.py --input=input.txt
```
