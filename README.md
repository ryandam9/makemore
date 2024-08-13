# Makemore

- Source: https://www.youtube.com/watch?v=t3YJ5hKiMQ0&t=5s
- [Colab](https://colab.research.google.com/drive/1CXVEmCO_7r7WYZGb5qnjfyxTvQa13g5X?usp=sharing)

## 1. Train the model

```sh
cd makemore

# Create a virtual env & install the dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python makemore_1_train_model.py
python makemore_2_inference.py
python makemore_3_export_to_onnx_dynamo.py
python makemore_4_inference_onnx.py
```

## 2. Note

As of Aug, 2024, when using `torch.onnx.dynamo_export` to export a PyTorch model to ONNX format, getting this error:

```sh
UserWarning: torch.onnx.dynamo_export only implements opset version 18 for now. If you need to use a different opset version, please register them with register_custom_op.
```

- This page talks about package versions & Opset version compatibility: [Compatability matrix](https://onnxruntime.ai/docs/reference/compatibility.html)

If I try to use latest version of PyTorch, I get the following error:

```sh
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Node (MatMulBnFusion_Gemm) Op (Gemm) [ShapeInferenceError] First input does not have rank 2
```

Just to be able to run error free, changed PyTorch version to 2.1.0.

---

## 3. Next steps

I like to run this model in a browser. At this point, I am yet to find out how to use `softmax()` and `multinomial()` in the browser. Does ONNX web runtime provide these?
