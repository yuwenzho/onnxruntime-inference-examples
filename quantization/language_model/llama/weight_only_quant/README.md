Step-by-Step
============

This example confirms INT4 weight only quantization (WOQ) accuracy for Llama-2 models on [lambada_openai](https://huggingface.co/datasets/EleutherAI/lambada_openai).

As large language models (LLMs) become more prevalent, there is a growing need for new and improved quantization methods that can meet the computational demands of these modern architectures while maintaining the accuracy. Compared to normal quantization like W8A8, weight only quantization is probably a better trade-off to balance the performance and the accuracy.

Two weight only algorithms are provided in this example. Round-to-nearest (RTN) is the most straightforward way to quantize weight using scale maps. GPTQ algorithm provides more accurate quantization but requires more computational resources.

# Prerequisite

## 1. Environment
```shell
pip install -r requirements.txt
```

## 2. Prepare Model

Note that this README.md uses meta-llama/Llama-2-7b-hf as an example. There are other models available that can be used for INT4 weight only quantization. The following table shows a few models' configurations:

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
optimum-cli export onnx --model meta-llama/Llama-2-7b-hf --task text-generation-with-past ./Llama-2-7b-hf
```

> Note: Llama-2-70b and Llama-2-70b-chat-hf will fail during export because of the different amounts of past keys/values and Attention. Set `--task text-generation` to disable the export with past keys/values reuse.

# Run

## 1. Quantization

```bash
bash run_quant.sh --input_model=/path/to/model \ # folder path of onnx model
                  --output_model=/path/to/model_quant \ # folder path to save onnx model
                  --batch_size=batch_size  \ # optional
                  --dataset=NeelNanda/pile-10k \
                  --tokenizer=meta-llama/Llama-2-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                  --algorithm=RTN # support RTN, GPTQ
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \ # folder path of onnx model
                      --batch_size=batch_size \ # optional 
                      --tokenizer=meta-llama/Llama-2-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                      --tasks=lambada_openai
```

# Accuracy results

The following table shows the accuracy results of Llama-2 models evaluated on lambada_openai task. `GPTQ W4G32Asym` in Configuration column means GPTQ algorithm is used for 4-bit weight only quantization, setting group_size=32 and scheme=asym.

<table class="tg">
<thead>
  <tr>
    <th rowspan="2">Model name</th>
    <th rowspan="2">Configuration</th>
    <th colspan="2">Lambada_openai</th>
    <th rowspan="2">Accuracy Ratio<br>[WOQ/FP32]</th>
    <th rowspan="2">Huggingface link</th>
  </tr>
  <tr>
    <th>Accuracy</th>
    <th>Perplexity</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">meta-llama/Llama-2-7b-chat-hf</td>
    <td>FP32</td>
    <td>0.7058</td>
    <td>3.2788</td>
    <td>/</td>
    <td>https://huggingface.co/meta-llama/Llama-2-7b-chat-hf</td>
  </tr>
  <tr>
    <td>GPTQ<br>W4G32Asym</td>
    <td>0.7025</td>
    <td>3.4489</td>
    <td>99.53%</td>
    <td>https://huggingface.co/Intel/Llama-2-7b-chat-hf-onnx-int4</td>
  </tr>
  <tr>
    <td rowspan="2">meta-llama/Llama-2-7b-hf</td>
    <td>FP32</td>
    <td>0.7392</td>
    <td>3.3950</td>
    <td>/</td>
    <td>https://huggingface.co/meta-llama/Llama-2-7b-hf</td>
  </tr>
  <tr>
    <td>GPTQ<br>W4G32Asym</td>
    <td>0.7326</td>
    <td>3.5286</td>
    <td>99.11%</td>
    <td>https://huggingface.co/Intel/Llama-2-7b-hf-onnx-int4</td>
  </tr>
  <tr>
    <td rowspan="2">meta-llama/Llama-2-13b-chat-hf</td>
    <td>FP32</td>
    <td>0.7312</td>
    <td>2.9163</td>
    <td>/</td>
    <td>https://huggingface.co/meta-llama/Llama-2-13b-chat-hf</td>
  </tr>
  <tr>
    <td>GPTQ<br>W4G128Asym</td>
    <td>0.7289</td>
    <td>3.0061</td>
    <td>99.68%</td>
    <td>https://huggingface.co/Intel/Llama-2-13b-chat-hf-onnx-int4</td>
  <tr>
    <td rowspan="2">meta-llama/Llama-2-13b-hf</td>
    <td>FP32</td>
    <td>0.7677</td>
    <td>3.0438</td>
    <td>/</td>
    <td>https://huggingface.co/meta-llama/Llama-2-13b-hf</td>
  </tr>
  <tr>
    <td>GPTQ<br>W4G32Asym</td>
    <td>0.7607</td>
    <td>3.1562</td>
    <td>99.09%</td>
    <td>https://huggingface.co/Intel/Llama-2-13b-hf-onnx-int4</td>
  </tr>
  <tr>
    <td rowspan="2">meta-llama/Llama-2-70b-chat-hf</td>
    <td>FP32</td>
    <td>0.7543</td>
    <td>2.6181</td>
    <td>/</td>
    <td>https://huggingface.co/meta-llama/Llama-2-70b-chat-hf</td>
  </tr>
  <tr>
    <td>RTN<br>W4G32Asym</td>
    <td>0.7489</td>
    <td>2.6850</td>
    <td>99.28%</td>
    <td>https://huggingface.co/Intel/Llama-2-70b-chat-hf-onnx-int4</td>
  </tr>
  <tr>
    <td rowspan="2">meta-llama/Llama-2-70b-hf</td>
    <td>FP32</td>
    <td>0.7964</td>
    <td>2.6612</td>
    <td>/</td>
    <td>https://huggingface.co/meta-llama/Llama-2-70b-hf</td>
  </tr>
  <tr>
    <td>RTN<br>W4G32Sym</td>
    <td>0.7896</td>
    <td>2.7546</td>
    <td>99.15%</td>
    <td>https://huggingface.co/Intel/Llama-2-70b-hf-onnx-int4</td>
  </tr>
</tbody>
</table>

> Note: The above results are obtained using `onnxruntime 1.16.0` and Intel® Neural Compressor built from master branch. Weight-only quantization in Intel® Neural Compressor is still under development. We encourage you to use the master branch to access the latest features.