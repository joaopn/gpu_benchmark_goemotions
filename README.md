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


| GPU/batch size        |    1   |      2     |      4     |      8      |    16   |   32   |
|-----------------------|:------:|:----------:|:----------:|:-----------:|:-------:|:------:|
| NVIDIA H100 (onnx)    | 206.33 |   668.66   |   938.49   | **1118.24** | 1111.50 | 916.40 |
| NVIDIA H100 (torch) | 233.93 |   409.74   |   643.75   |  **761.87** |  709.18 | 583.89 |
| RTX 4090 (onnx)       | 529.99 |   806.25   | **995.21** |    973.02   |  789.57 | 582.33 |
| RTX 4090 (torch)    | 305.37 |   526.76   | **770.99** |    764.37   |  599.78 | 435.20 |
| RTX 2080Ti (onnx)     | 259.95 | **308.22** |   307.64   |    260.16   |  202.73 | 154.96 |
| RTX 2080Ti (torch)  | 119.95 |   187.07   | **261.68** |    249.27   |  199.35 | 152.25 |
| Tesla P40 (onnx)      | 203.06 | **246.09** |   233.86   |    193.70   |  150.43 | 116.60 |
| Tesla P40 (torch)   | 152.85 |   201.74   | **207.14** |    170.94   |  121.95 |  79.43 |
| Tesla P100 (onnx)     | 183.48 | **227.88** |   220.31   |    180.86   |  141.63 | 113.99 |
| Tesla P100 (torch)  | 110.53 |   173.81   | **197.23** |    179.88   |  138.12 | 104.62 |

**Table 1:** GPU benchmark in messages/s for the normal dataset. Results may vary due to CPU tokenizer performance.
</details>

<details>
<summary>GPU results for the filtered (>200 characters) dataset</summary>

| GPU/batch size        |      1     |      2     |      4     |      8     |   16   |   32   |
|-----------------------|:----------:|:----------:|:----------:|:----------:|:------:|:------:|
| NVIDIA H100 (onnx)    |   365.26   |   545.62   |   694.03   | **750.14** | 698.55 | 575.51 |
| NVIDIA H100 (torch) |   216.78   |   345.62   |   444.69   | **451.34** | 401.60 | 370.30 |
| RTX 4090 (onnx)       |   443.78   |   585.20   | **631.64** |   542.55   | 436.59 | 358.92 |
| RTX 4090 (torch)    |   286.64   |   437.60   | **472.70** |   397.95   | 315.54 | 260.67 |
| RTX 2080Ti (onnx)     |   171.54   | **180.66** |   164.71   |   137.18   | 113.08 |  98.17 |
| RTX 2080Ti (torch)  |   111.47   |   155.08   | **155.40** |   132.38   | 110.22 |  95.87 |
| Tesla P40 (onnx)      |   134.29   | **139.11** |   122.22   |   100.71   |  84.91 |  73.84 |
| Tesla P40 (torch)   |   108.46   | **118.71** |   107.99   |    85.34   |  63.67 |  47.16 |
| Tesla P100 (onnx)     | **132.62** |   131.18   |   117.33   |    97.76   |  82.37 |  73.50 |
| Tesla P100 (torch)  |    93.73   | **111.95** |   111.82   |    96.39   |  77.67 |  66.77 |

**Table 2:** GPU benchmark in messages/s for the filtered dataset. Results may vary due to CPU tokenizer performance.
</details>

<details>
<summary>CPU results for the normal dataset</summary>

| CPU/batch size @threads    |   1 @1T   | 2 @1T | 4 @1T | 1 @4T | 2 @4T | 4 @4T | @max cores* |
|----------------------------|:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|
| Ryzen 7800X3D 8C (onnx)    | **15.67** | 10.60 |  7.14 | 13.66 |  9.74 |  6.45 |    125.36   |
| Ryzen 7800X3D 8C (torch) | **11.96** |  9.39 |  6.64 |  7.61 |  6.51 |  4.70 |    95.68    |
| Ryzen 5900X 12C (onnx)     | **14.29** |  9.96 |  6.79 | 10.58 |  8.13 |  5.83 |    171.48   |
| Ryzen 5900X 12C (torch)  | **11.14** |  9.02 |  6.33 |  6.73 |  5.85 |  4.29 |    133.68   |
| Ryzen 3600 6C (onnx)       | **12.28** |  8.66 |  5.88 |  8.74 |  6.86 |  4.96 |    73.68    |
| Ryzen 3600 6C (torch)    |  **8.12** |  6.35 |  4.43 |  4.93 |  4.16 |  3.00 |    48.72    |
| 2x Epyc 7702 64C (onnx)    |  **9.27** |  6.81 |  4.75 |  5.33 |  4.25 |  3.18 |   1186.56   |
| 2x Epyc 7702 64C (torch) |  **5.57** |  4.73 |  3.37 |  3.12 |  2.79 |  2.15 |    712.96   |
| 2x Xeon 4214 24C (onnx)    |  **6.64** |  4.39 |  2.91 |  5.33 |  3.60 |  2.42 |    318.72   |
| 2x Xeon 4214 24C (torch) |  **4.57** |  3.74 |  2.65 |  2.76 |  2.80 |  2.18 |    219.36   |

**Table 3:** CPU benchmark in **messages/thread/s**. *(@max cores) = (performance @1T)x(number of cores). It underestimates performance by disregarding hyperthreading, but overestimates by assuming same frequency at single-threaded and full load. 
</details>

<details>
<summary>CPU results for the filtered dataset</summary>

| CPU/batch size @threads    |   1 @1T  | 2 @1T | 4 @1T | 1 @4T | 2 @4T | 4 @4T | @max cores* |
|----------------------------|:--------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|
| Ryzen 5900X 12C (onnx)     | **5.61** |  4.19 |  3.09 |  4.51 |  3.58 |  2.69 |    67.32    |
| Ryzen 5900X 12C (torch)  | **4.99** |  3.94 |  2.85 |  3.27 |  2.69 |  1.96 |    59.88    |
| Ryzen 3600 6C (onnx)       | **4.90** |  3.67 |  2.68 |  3.80 |  3.03 |  2.33 |     29.4    |
| Ryzen 3600 6C (torch)    | **3.57** |  2.75 |  2.01 |  2.30 |  1.85 |  1.36 |    21.42    |
| 2x Epyc 7702 64C (onnx)    | **3.82** |  2.94 |  2.18 |  2.17 |  1.89 |  1.66 |    488.96   |
| 2x Epyc 7702 64C (torch) | **2.64** |  2.14 |  1.54 |  1.55 |  1.30 |  1.04 |    337.92   |

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