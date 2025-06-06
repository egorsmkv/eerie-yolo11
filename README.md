# IREE YOLOv11

My try to run YOLOv11 on IREE with https://github.com/gmmyung/eerie.

## Create a virtual environment

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```
uv venv --python 3.12
source .venv/bin/activate
```

## Install IREE compiler and runtime

```
python -m ensurepip --upgrade
python -m pip install --upgrade pip

python -m pip install iree-base-compiler==3.3.0 iree-base-runtime==3.3.0 onnx==1.17.0
```

## Set cargo flags

```
python create_cargo_config.py
```

## Download an ONNX version of YOLOv11

```
wget "https://huggingface.co/qualcomm/YOLOv11-Detection/resolve/af104380457eda0e213c91dcddc04d9011c41bb4/YOLOv11-Detection.onnx"
```

## Convert to MLIR

```
iree-import-onnx YOLOv11-Detection.onnx -o yolo11.mlir
```

## Introspect system

Devices:

```
iree-run-module --list_devices
```

Drives:

```
iree-run-module --list_drivers
```

## Compile MLIR for CPU

```
iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-cpu=host -o yolo11_cpu.vmfb yolo11.mlir
```

## Compile MLIR for CUDA

```
iree-compile --iree-stream-external-resources-mappable=true --iree-hal-target-device=cuda --iree-cuda-target=sm_60 -o yolo11_cuda.vmfb yolo11.mlir
```

## Test runtime

Run on CPU with zeroed tensor:

```
iree-run-module --device=local-task --module=yolo11_cpu.vmfb --input="1x3x640x640xf32=0"
```

## Build and Run

```
cargo build

RUST_LOG=trace ./target/debug/eerie-yolo11
```

---

## YOLOv11x

```
wget "https://huggingface.co/pan93412/yolo-v11-onnx/resolve/main/yolo11x.onnx"
```

```
iree-import-onnx yolo11x.onnx -o yolo11x.mlir

iree-compile --iree-stream-external-resources-mappable=true --iree-hal-target-device=cuda --iree-cuda-target=sm_75 -o yolo11x_cuda.vmfb yolo11x.mlir
```

