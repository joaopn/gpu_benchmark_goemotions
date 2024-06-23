# GPU Benchmark GoEmotions

GPU and CPU Benchmark of the [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions) model on a dataset of 10k random reddit comments, with pytorch ([`torch`](https://huggingface.co/SamLowe/roberta-base-go_emotions)), ONNX ([`onnx`](https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx)), and optimized FP16 ONNX versions ([`onnx-fp16`](https://huggingface.co/joaopn/roberta-base-go_emotions-onnx-fp16)). 

## Results
GPU insights:
- ONNX with CUDA is up to ~40% faster than torch. It can be optimized further with [TensorRT](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu#tensorrtexecutionprovider) or [model quantization](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/quantization)
- For reddit comments on the GPU, batch size 2 or 4 is usually fastest
- The RTX 4090 is 4-5X faster than a Tesla P40, but around 9X more expensive (~$1800 vs ~$200 used)
- The H100 performs 10-20% better than the 4090, but it is around 15X more expensive (thanks @ruggsea for the H100 and P100 numbers!)

CPU insights:
- ONNX is up to ~60% faster than torch.  It can be optimized further with [model quantization](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/)
- For reddit comments on the CPU, unbatched and single-threaded is always faster
- A fully loaded 2x Epyc 7702 64C (Zen2 Rome) system is about equivalent to the RTX 4090 (~20% faster on the normal dataset, ~20% slower on the filtered dataset)


### Benchmark results

<details open>

<summary>GPU results for the normal dataset</summary>


| GPU/batch size        |    1   |    2   |    4   |    8   |   16   |   32   |
|-----------------------|:------:|:------:|:------:|:------:|:------:|:------:|
| Tesla P40 (onnx-fp16) | 263.18 | 286.72 | 255.36 | 200.65 | 148.89 | 108.92 |
| Tesla P40 (onnx)      | 212.35 | 260.29 | 247.01 | 202.54 | 155.42 | 119.59 |
| Tesla P40 (torch)     | 162.19 | 218.12 | 221.68 | 177.85 | 124.72 |  80.36 |

**Table 1:** GPU benchmark in messages/s for the normal dataset. Results may vary due to CPU tokenizer performance.
</details>

<details>
<summary>GPU results for the filtered (>200 characters) dataset</summary>

| GPU/batch size        |    1   |    2   |    4   |    8   |   16  |   32  |
|-----------------------|:------:|:------:|:------:|:------:|:-----:|:-----:|
| Tesla P40 (onnx-fp16) | 154.33 | 150.74 | 126.01 | 101.90 | 81.77 | 68.15 |
| Tesla P40 (onnx)      | 138.25 | 142.59 | 125.45 | 103.09 | 86.84 | 75.27 |
| Tesla P40 (torch)     | 117.11 | 128.19 | 113.87 |  88.03 | 64.88 | 47.76 |

**Table 2:** GPU benchmark in messages/s for the filtered dataset. Results may vary due to CPU tokenizer performance.
</details>

<details>
<summary>CPU results for the normal dataset</summary>

| CPU/batch size @threads    |   1 @1T  | 2 @1T | 4 @1T | 1 @4T | 2 @4T | 4 @4T | @max cores* |
|----------------------------|:--------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|
|      |  |   |   |   |   |   |        |


**Table 3:** CPU benchmark in **messages/thread/s**. *(@max cores) = (performance @1T)x(number of cores). It underestimates performance by disregarding hyperthreading, but overestimates by assuming same frequency at single-threaded and full load. 
</details>

<details>
<summary>CPU results for the filtered dataset</summary>

| CPU/batch size @threads    |   1 @1T  | 2 @1T | 4 @1T | 1 @4T | 2 @4T | 4 @4T | @max cores* |
|----------------------------|:--------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|
|      |  |   |   |   |   |   |        |


**Table 4:** CPU benchmark in **messages/thread/s**. *(@max cores) = (performance @1T)x(number of cores). It underestimates performance by disregarding hyperthreading, but overestimates by assuming same frequency at single-threaded and full load. 
</details>

## Documentation

### Requirements

The benchmark requires a working `torch` installation with CUDA support, as well as `transformers`, `optimum`, `pandas` and `tqdm`. These can be installed with

```
pip install transformers optimum[onnxruntime-gpu] pandas tqdm --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
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

To run the benchmarks, use the `run_benchmark.py` script:

```
python run_benchmark.py --model [torch, onnx or onnx-fp16] --device [gpu or cpu]
```

Arguments:
- `model` (required): Model backend to use, either "torch" for torch or "onnx" for ONNX Runtime.
- `device` (required): Device type to use, either "gpu" or "cpu"
- `dataset`: Dataset variant to use, either "normal" or "filtered" (default: "normal").
- `gpu`: ID of the GPU to use (default: 0).
- `batches`: Comma-separated batch sizes to run (default: "1,2,4,8,16,32").
- `threads`: Specify the number of CPU threads to use (default: 1).

To run the CPU benchmarks, use the `run_benchmark_cpu.py` script:

```
python run_benchmark_cpu.py --model [torch or onnx] --threads [number of threads]
```

Arguments:
- `model`: Specify the model backend to use, either "torch" for torch or "onnx" for ONNX Runtime (default: "torch").
- `dataset`: Specify the dataset to use, either "normal" or "filtered" (default: "normal").
- `threads`: Specify the number of CPU threads to use (default: 1).
- `batches`: Specify the comma-separated batch sizes to run (default: "1,2,4").

The scripts will output the number of messages processed per second for each batch size.