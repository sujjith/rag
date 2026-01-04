# Phase 08: Advanced Topics

**Duration**: 3 weeks | **Prerequisites**: Phase 07 completed

---

## Week 1: AI Agents

### LangGraph
```bash
uv add langgraph
```

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    next_action: str

def router(state: AgentState):
    if "search" in state["messages"][-1]:
        return "search_node"
    return "respond_node"

graph = StateGraph(AgentState)
graph.add_node("router", router)
graph.add_node("search_node", search_tool)
graph.add_node("respond_node", respond)
graph.set_entry_point("router")
graph.add_edge("search_node", END)
graph.add_edge("respond_node", END)

app = graph.compile()
```

### AutoGen
```bash
uv add pyautogen
```

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config={"model": "gpt-4"})
user_proxy = UserProxyAgent("user", human_input_mode="NEVER")
user_proxy.initiate_chat(assistant, message="Write a Python function...")
```

---

## Week 2: Model Optimization

### ONNX
```bash
uv add onnx onnxruntime
```

```python
import torch
import onnx

# Export PyTorch to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# Run with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
result = session.run(None, {"input": input_data})
```

### Quantization
```bash
uv add bitsandbytes
```

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    load_in_8bit=True,
    device_map="auto"
)
```

---

## Week 3: Privacy & Explainability

### SHAP
```bash
uv add shap
```

```python
import shap
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])
```

### Flower (Federated Learning)
```bash
uv add flwr
```

```python
import flwr as fl

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train)
        return model.get_weights(), len(x_train), {}

fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
```

---

## Milestone Checklist
- [ ] LangGraph agent built
- [ ] AutoGen multi-agent working
- [ ] ONNX model exported
- [ ] Model quantized
- [ ] SHAP explanations generated
- [ ] Federated learning simulated

**Next**: [Phase 09](./phase_09_projects.md)
