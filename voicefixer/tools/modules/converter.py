# models/converter.py

import torch
from core.vocoder import VoiceFixerVocoder

def export_onnx(model_path="voicefixer.onnx"):
    model = VoiceFixerVocoder().model  # Extract Generator
    dummy_input = torch.randn(1, 1, 80, 128)  # (batch, channels, time, mel)
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        input_names=["mel"],
        output_names=["audio"],
        dynamic_axes={"mel": {2: "time"}},
        opset_version=13
    )
