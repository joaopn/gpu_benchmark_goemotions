# GPU Benchmark GoEmotions

GPU and CPU Benchmark of the [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions) model on a dataset of 10k random reddit comments, with pytorch ([`torch`](https://huggingface.co/SamLowe/roberta-base-go_emotions)), ONNX ([`onnx`](https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx)), and O4-optimized FP16 ONNX versions ([`onnx-fp16`](https://huggingface.co/joaopn/roberta-base-go_emotions-onnx-fp16)). 

## Results
GPU insights:
- The FP16 optimized model is up to **3X** faster than torch. The gain depends on the GPU's specific FP32:FP16 ratio.
- Base ONNX with CUDA is up to ~40% faster than torch. In theory, it can be optimized further with [TensorRT](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu#tensorrtexecutionprovider).
- The RTX 4090 is both 9X faster and 9X more expensive than the P40 (~$1800 vs ~$200 used) with FP16 ONNX, and only 4-5X faster with the other models.


### Benchmark results

<details open>

<summary>GPU results for the normal dataset</summary>


| GPU/batch size         |    1    |      2     |      4      |      8      |    16   |    32   |
|------------------------|:-------:|:----------:|:-----------:|:-----------:|:-------:|:-------:|
| RTX 4090 (onnx-fp16)   | 1042.47 |   1042.47  |   2280.61   | **2551.59** | 2346.59 | 2346.59 |
| RTX 4090 (onnx)        |  595.40 |   963.06   | **1232.12** |   1183.82   |  919.05 |  646.79 |
| RTX 4090 (torch)       |  323.75 |   564.39   |    857.28   |  **876.10** |  668.70 |  462.63 |
| Tesla A10G (onnx-fp16) |  600.00 |   879.20   | **1094.11** |   1082.87   |  943.09 |  767.02 |
| Tesla A10G (onnx)      |  326.58 |   476.80   |  **556.52** |    473.00   |  365.13 |  281.95 |
| Tesla A10G (torch)     |  131.10 |   236.48   |    385.63   |  **402.36** |  310.15 |  231.54 |
| Tesla P40 (onnx-fp16)  |  263.18 | **286.72** |    255.36   |    200.65   |  148.89 |  108.92 |
| Tesla P40 (onnx)       |  212.35 | **260.29** |    247.01   |    202.54   |  155.42 |  119.59 |
| Tesla P40 (torch)      |  162.19 |   218.12   |  **221.68** |    177.85   |  124.72 |  80.36  |

**Table 1:** GPU benchmark in messages/s for the normal dataset. Results may vary due to CPU tokenizer performance.
</details>

<details>
<summary>GPU results for the filtered (>200 characters) dataset</summary>

| GPU/batch size         |    1   |      2     |      4     |      8      |    16   |    32   |
|------------------------|:------:|:----------:|:----------:|:-----------:|:-------:|:-------:|
| RTX 4090 (onnx-fp16)   | 856.65 |   1209.98  |   1438.25  | **1513.05** | 1395.42 | 1221.52 |
| RTX 4090 (onnx)        | 494.28 |   673.83   | **740.03** |    610.06   |  472.35 |  382.72 |
| RTX 4090 (torch)       | 302.38 |   476.46   | **548.32** |    450.82   |  338.37 |  273.01 |
| Tesla A10G (onnx-fp16) | 463.21 |   584.19   | **624.32** |    612.12   |  554.00 |  498.06 |
| Tesla A10G (onnx)      | 255.55 | **312.77** |   290.70   |    239.00   |  200.90 |  176.20 |
| Tesla A10G (torch)     | 126.82 |   209.08   | **245.60** |    205.70   |  167.53 |  141.90 |
| Tesla P40 (onnx-fp16)  | 154.33 | **150.74** |   126.01   |    101.90   |  81.77  |  68.15  |
| Tesla P40 (onnx)       | 138.25 | **142.59** |   125.45   |    103.09   |  86.84  |  75.27  |
| Tesla P40 (torch)      | 117.11 | **128.19** |   113.87   |    88.03    |  64.88  |  47.76  |

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


The scripts will output the number of messages processed per second for each batch size.