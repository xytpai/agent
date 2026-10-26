## Agent

A lightweight ReAct agent that combines language models with tool execution to tackle tasks through iterative reasoning and action, with persistent conversation history.

### Backend Test

```bash
export BASE_URL=${YOUR_LLM_GATEWAY_URL}
export API_KEY=${YOUR_LLM_GATEWAY_API_KEY}
export MODEL_NAME=${YOUR_LLM_MODEL_NAME}
export REASONING_EFFORT=high

python core/backends.py "Introduce yourself"
```

### Run agent

```bash
python core/agent.py --input=input.txt --history=new 2>&1 | tee log
```

### Web server

```bash
export AGENT_WEB_USERNAME="your-username"
export AGENT_WEB_PASSWORD="your-password"
python server.py \
  --host 127.0.0.1 \
  --port 5500 \
  --ssl-cert "$SERVER_PREFIX/.local/state/agent-certificates/config/live/www.xytpai.top/fullchain.pem" \
  --ssl-key "$SERVER_PREFIX/.local/state/agent-certificates/config/live/www.xytpai.top/privkey.pem"
```
