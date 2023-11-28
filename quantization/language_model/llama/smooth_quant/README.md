Step-by-Step
============

This folder contains example code for quantizing LLaMa model.

# Prerequisite

## 1. Environment
```shell
pip install -r requirements.txt
```

## 2. Prepare Model

Note that this README.md uses meta-llama/Llama-2-7b-hf as an example. There are other models available that can be used for SmoothQuant. The following table shows a few models' configurations:

| Model | Num Hidden Layers| Num Attention Heads | Hidden Size |
| --- | --- | --- | --- |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 32 | 32 | 4096 |
| [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | 32 | 32 | 4096 |
| [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf) | 40 | 40 | 5120 |
| [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | 40 | 40 | 5120 |
| [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf) | 80 | 64 | 8192 |
| [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) | 80 | 64 | 8192 |

Export to ONNX model:

```bash
optimum-cli export onnx --model meta-llama/Llama-2-7b-hf --task text-generation-with-past --legacy ./Llama-2-7b-hf
```

# Run

## 1. Quantization

```bash
bash run_quantization.sh --input_model=/path/to/model \ # folder path of onnx model
                         --output_model=/path/to/model_tune \ # folder path to save onnx model
                         --model_name_or_path=meta-llama/Llama-2-7b-hf \ # huggingface model id or folder path containing tokenizer and config file
                         --batch_size=batch_size # optional \
                         --dataset NeelNanda/pile-10k \
                         --alpha 0.6 \ 
                         --quant_format="QOperator" # or QDQ, optional
```

## 2. Benchmark

Accuracy:

```bash
bash run_benchmark.sh --input_model=path/to/model \ # folder path of onnx model
                      --model_name_or_path=meta-llama/Llama-2-7b-hf \ # huggingface model id or folder path containing tokenizer and config file
                      --batch_size=batch_size \ # optional 
                      --mode=accuracy \
                      --tasks=lambada_openai
```

Performance:
```bash
numactl -m 0 -C 0-3 bash run_benchmark.sh --input_model=path/to/model \ # folder path of onnx model
                                          --model_name_or_path=meta-llama/Llama-2-7b-hf \ # huggingface model id or folder path containing tokenizer and config file
                                          --mode=performance \
                                          --batch_size=batch_size # optional \
                                          --intra_op_num_threads=4
```
