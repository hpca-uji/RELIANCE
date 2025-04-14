import torch
from torch import jit
from torch import onnx
from onnxruntime import InferenceSession
from numpy.testing import assert_allclose


def export_model(
    model, save_path, save_format, input_size=None, subformat=None, opset=None
):
    """Exports model given certain parameters, calling internal exporters

    Args:
    model: Model to be exported
    input_size: Dimensions of model input
    format: Format of the exported model
    subformat: Subformat used in case of jit or onnx formats (script|trace)
    save_pth: Path where model will be saved
    opset: Opset used in onnx format
    """

    if save_format == "torch":
        _export_model_torch(model, save_path)
    elif save_format == "jit":
        _export_model_jit(model, save_path, input_size, subformat)
    elif save_format == "onnx":
        _export_model_onnx(model, save_path, input_size, subformat, opset)
    else:
        raise ValueError(f"{save_format} not supported as save format.")


def _export_model_subformat(model, input_size, subformat):
    """Subformat exporters (Script and Trace)"""

    # Create dummy input
    dummy_input = torch.rand(input_size)

    # Check subformat parameter
    if subformat == "script":
        jitmodel = jitmodel = jit.script(model, example_inputs=[dummy_input])
    elif subformat == "trace":
        jitmodel = jit.trace(model, dummy_input)
    else:
        raise ValueError(f"{subformat} not supported as save subformat.")

    # Assert validity
    output0 = model(dummy_input).detach().numpy()
    output1 = jitmodel(dummy_input).detach().numpy()
    assert_allclose(output0, output1, rtol=1e-03, atol=1e-05)

    return jitmodel, dummy_input


def _export_model_torch(model, save_path):
    """Torch exporter"""
    torch.save([model.kwargs, model.state_dict()], save_path)


def _export_model_jit(model, save_path, input_size, subformat):
    """Jit exporter"""
    jitmodel, _ = _export_model_subformat(model, input_size, subformat)
    jit.save(jitmodel, save_path)


def _export_model_onnx(model, save_path, input_size, subformat, opset):
    """ONNX exporter"""
    jitmodel, dummy_input = _export_model_subformat(model, input_size, subformat)

    # Onnx export
    input_names = ["input_0"]
    output_names = ["output_0"]
    onnx.export(
        jitmodel,
        dummy_input,
        save_path,
        opset_version=opset,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        dynamic_axes={
            "input_0": {0: "batch_size", 2: "image_width", 3: "image_height"},
            "output_0": {0: "batch_size"},
        },
    )

    # Assert validity
    output0 = jitmodel(dummy_input).detach().numpy()
    onnx_session = InferenceSession(save_path)
    onnx_input = {onnx_session.get_inputs()[0].name: dummy_input.numpy()}
    output1 = onnx_session.run(None, onnx_input)[0]
    assert_allclose(output0, output1, rtol=1e-03, atol=1e-05)
