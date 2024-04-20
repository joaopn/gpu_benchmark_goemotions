# GPU Benchmark GoEmotions

GPU and CPU Benchmark of the [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions) model on a dataset of 10k random reddit comments, with both [`pytorch`]((https://huggingface.co/SamLowe/roberta-base-go_emotions)) and [`ONNX`](https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx) versions. 

### Requirements

The benchmark requires a working `pytorch` installation with CUDA support, as well as `transformers`, `optimum`, `onnxruntime`, `pandas` and `tqdm`. These can be installed with

```
pip install transformers[onnx] optimum[onnxruntime-gpu] onnxruntime-gpu pandas tqdm --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

Alternatively, a conda environment `bench` with all the requirements can be created with

```
conda env create -f environment.yml
conda activate bench
```
### Dataset

The dataset consists of 10K randomly sampled Reddit comments from 12/2005-03/2023, from the [Pushshift data dumps](https://academictorrents.com/details/9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4). It excludes comments with empty, `[deleted]` or `[removed]` content. Two options are provided:
- `normal`: As described above
- `filtered`: contains only comments with `>200` characters. 


### Usage

To run the GPU benchmarks, use the `run_benchmark.py` script:

```
python run_benchmark.py --model [pytorch or onnx]
```

Arguments:
- `model` (required): Model backend to use, either "pytorch" for PyTorch or "onnx" for ONNX Runtime.
- `dataset`: Dataset variant to use, either "normal" or "filtered" (default: "normal").
- `gpu_id`: ID of the GPU to use (default: 0).
- `batches`: Comma-separated batch sizes to run (default: "1,4,8,16,32").

To run the CPU benchmarks, use the `run_benchmark_cpu.py` script:

```
python run_benchmark_cpu.py --model [pytorch or onnx] --threads [number of threads]
```

Arguments:
- `model`: Specify the model backend to use, either "pytorch" for PyTorch or "onnx" for ONNX Runtime (default: "pytorch").
- `dataset`: Specify the dataset to use, either "normal" or "filtered" (default: "normal").
- `threads`: Specify the number of CPU threads to use (default: 1).
- `batches`: Specify the comma-separated batch sizes to run (default: "1,4,8,16,32").

The scripts will output the number of messages processed per second for each batch size.