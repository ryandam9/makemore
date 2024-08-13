"""
This script exports the makemore model to ONNX format.

Ref:
https://pytorch.org/docs/stable/onnx_dynamo.html#a-simple-example
"""
import os.path

import onnx
import onnxruntime
import torch


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def print_nodes():
    model = onnx.load(onnx_model)

    # Print out all the nodes in the model
    for node in model.graph.node:
        print(f"Node: {node.name}, Op Type: {node.op_type}")

        for input in node.input:
            print(f"  Input: {input}")

        for output in node.output:
            print(f"  Output: {output}")

        print()


def verify_onnx_file():
    model = onnx.load(onnx_model)

    # Check that the model is well-formed
    onnx.checker.check_model(model)

    # Print a human-readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))


def inspect_onnx_model(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    print("Model inputs:")
    for input in model.graph.input:
        print(f"  Name: {input.name}")
        print(f"  Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")
        print(f"  Type: {input.type.tensor_type.elem_type}")
        print()

    print("Model outputs:")
    for output in model.graph.output:
        print(f"  Name: {output.name}")
        print(
            f"  Shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}"
        )
        print(f"  Type: {output.type.tensor_type.elem_type}")
        print()

    print("Model operations:")
    for i, node in enumerate(model.graph.node):
        print(f"Node {i}:")
        print(f"  Op Type: {node.op_type}")
        print(f"  Name: {node.name}")
        print(f"  Inputs: {node.input}")
        print(f"  Outputs: {node.output}")

        # If it's a Gemm operation, let's check its attributes
        if node.op_type == "Gemm":
            print("  Gemm Attributes:")
            for attr in node.attribute:
                print(
                    f"    {attr.name}: {attr.i if attr.type == onnx.AttributeProto.INT else attr.f}"
                )
        print()

    # Check for initializers (weights and biases)
    print("Model initializers:")
    for initializer in model.graph.initializer:
        print(f"  Name: {initializer.name}")
        print(f"  Shape: {initializer.dims}")
        print(f"  Data type: {initializer.data_type}")
        # Uncomment the next line if you want to see the actual values
        # print(f"  Values: {numpy_helper.to_array(initializer)}")
        print()


models_location = "./models"
pytorch_model = os.path.join(models_location, "makemore_model.pth")
onnx_model = os.path.join(models_location, "makemore.onnx")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load PyTorch model
model = torch.load(pytorch_model, weights_only=False)
model = model.to(device)
model.eval()

# 2. Export the model to ONNX format
context = [0] * 8
torch_input = torch.tensor([context])
onnx_program = torch.onnx.dynamo_export(model, torch_input)
onnx_program.save(onnx_model)
print(f"Model exported to ONNX format: {onnx_model}")

print("\nNodes")
print("-" * 100)
print_nodes()

print("\nGraph")
print("-" * 100)
verify_onnx_file()

print("\nInputs, Outputs, Operations, Initializers")
print("-" * 100)
inspect_onnx_model(onnx_model)

# 3. Test the ONNX model with ONNX Runtime
onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)
print(f"Input length: {len(onnx_input)}")
print(f"Sample input to the model: {onnx_input}")

ort_session = onnxruntime.InferenceSession(
    onnx_model, providers=["CPUExecutionProvider"]
)

onnxruntime_input = {
    k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)
}

onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
print(f"Output from the model: {onnxruntime_outputs}")
