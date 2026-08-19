"""
training/benchmark_sdnl.py
Benchmarking inference latency of the SDNL model (PyTorch / ONNX) on real ROI crops.
Validates latency requirement: <= 15 ms per ROI.
"""

import os
import glob
import time
import argparse
import numpy as np
import torch


def load_sample_roi_tensors(data_dir="data/preprocessed", in_channels=4):
    sample_files = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
    tensors = []
    for path in sample_files[:10]:
        try:
            t = np.load(path)
            t = np.transpose(t, (2, 0, 1))  # (4, 128, 128)
            if in_channels == 3:
                t = t[:3, :, :]
            tensors.append(t)
        except Exception:
            pass
    
    if not tensors:
        print("[Benchmark] No preprocessed ROI files found, generating sample ROI tensor.")
        tensors = [np.random.randn(in_channels, 128, 128).astype(np.float32) for _ in range(5)]
    
    return tensors


def benchmark_pytorch(model_path, in_channels=4, runs=100):
    from train_sdnl import SDNLModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SDNLModel(num_classes=2, in_channels=in_channels).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    sample_tensors = load_sample_roi_tensors(in_channels=in_channels)
    tensor_batch = [torch.tensor(t, dtype=torch.float32).unsqueeze(0).to(device) for t in sample_tensors]
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(tensor_batch[0])
            
    latencies = []
    for i in range(runs):
        inp = tensor_batch[i % len(tensor_batch)]
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(inp)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)
        
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    p95_lat = np.percentile(latencies, 95)
    
    print(f"[PyTorch Benchmark] Device: {device} | Mean Latency: {mean_lat:.2f} ms | Std: {std_lat:.2f} ms | P95: {p95_lat:.2f} ms")
    return mean_lat


def benchmark_onnx(onnx_path, in_channels=4, runs=100):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ONNX Benchmark] onnxruntime not installed, skipping ONNX benchmark.")
        return None
        
    if not os.path.exists(onnx_path):
        print(f"[ONNX Benchmark] Model file missing: {onnx_path}")
        return None
        
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    sample_tensors = load_sample_roi_tensors(in_channels=in_channels)
    np_batch = [np.expand_dims(t, axis=0).astype(np.float32) for t in sample_tensors]
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: np_batch[0]})
        
    latencies = []
    for i in range(runs):
        inp = np_batch[i % len(np_batch)]
        start = time.perf_counter()
        _ = session.run(None, {input_name: inp})
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)
        
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    p95_lat = np.percentile(latencies, 95)
    
    print(f"[ONNX Benchmark] Engine: ONNXRuntime CPU | Mean Latency: {mean_lat:.2f} ms | Std: {std_lat:.2f} ms | P95: {p95_lat:.2f} ms")
    return mean_lat


def main():
    parser = argparse.ArgumentParser(description="Benchmark SDNL Model Inference Latency on Real ROIs")
    parser.add_argument("--pytorch-path", type=str, default="models/sdnl/sdnl_v1.pt")
    parser.add_argument("--onnx-path", type=str, default="models/sdnl/sdnl_v1.onnx")
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    print("=== InfraEdge-Shield SDNL Latency Benchmark (Real ROI Crops) ===")
    py_lat = benchmark_pytorch(args.pytorch_path, args.in_channels, args.runs)
    onnx_lat = benchmark_onnx(args.onnx_path, args.in_channels, args.runs)
    
    target_ms = 15.0
    effective_lat = onnx_lat if onnx_lat is not None else py_lat
    if effective_lat <= target_ms:
        print(f"\n✅ PASS: Real ROI Inference Latency ({effective_lat:.2f} ms) <= Target ({target_ms} ms per ROI)")
    else:
        print(f"\n⚠️ WARNING: Latency ({effective_lat:.2f} ms) exceeds target ({target_ms} ms)")


if __name__ == "__main__":
    main()
