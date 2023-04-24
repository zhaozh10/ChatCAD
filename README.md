# [ChatCAD+: Towards a Reliable and Universal Interactive CAD using LLMs](https://arxiv.org/abs/2302.07257)

by Zihao Zhao\*, Sheng Wang\*, Jinchen Gu*,
Yitao Zhu*, Lanzhuju Mei,
Zixu Zhuang, Zhiming Cui, Qian Wang, Dinggang Shen

[![arXiv](https://img.shields.io/badge/📃-arXiv-ff69b4)](https://arxiv.org/abs/2302.07257)

<!-- ![webpage](https://img.shields.io/badge/🖥-Website-9cf) -->

<div align="center">
  <img src="imgs/overview.png">
</div>

## Introduction

This repository provides the official implementation of some components of ChatCAD+:<br/>

- Modality identification <a src="https://colab.research.google.com/assets/colab-badge.svg" href="https://colab.research.google.com/drive/1mbBgkoyk4n_qAJasY5_cOAqg7I5WP1H7?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
- Interactive CAD of Chest X-rays
- LLM-based knowledge retrieval
- An easy-deploy local web ui (modified from [Gpt4All Web UI](https://github.com/ParisNeo/Gpt4All-webui.git) )

<!-- **[ChatCAD: Interactive Computer-Aided Diagnosis on Medical Image using Large Language Models](https://arxiv.org/abs/2302.07257)** <br/> -->

## 最近更新

- <img src="https://img.shields.io/badge/Version-0.0.3--alpha-brightgreen">(2023.4.18): P-Tuning & 多轮对话 & 模型可靠性提升

## 训练数据

| Dataset          | Department                | Language | Q&A | Chat | Number | Syn. | Size  | Weight                                                                     |
| ---------------- | ------------------------- | -------- | --- | ---- | ------ | ---- | ----- | -------------------------------------------------------------------------- |
| CMD.             | Surgical                  | CN       | ✔   | ×    | 116K   | ×    | 52MB  |                                                                            |
|                  | Obstetrics and Gynecology | CN       | ✔   | ×    | 229K   | ×    | 78MB  |                                                                            |
|                  | Pediatrics                | CN       | ✔   | ×    | 117K   | ×    | 47MB  |                                                                            |
|                  | Internal Medicine         | CN       | ✔   | ×    | 307K   | ×    | 102MB |                                                                            |
|                  | Andriatria                | CN       | ✔   | ×    | 113K   | ×    | 44MB  |                                                                            |
|                  | Merged                    | CN       | ✔   | ×    | 1.9M   | ×    |       | Doctor_GLM/ckpt                                                            |
| MedDialog        | Multiple                  | CN&EN    | ✔   | ✔    | 3.4M   | ×    | 1.5GB | [ptuning_weight](https://pan.baidu.com/s/1Yf56egVGwI0XN2iOLcEGSQ?pwd=r4p0) |
| ChatDoctor       | Multiple                  | EN       | ✔   | ×    | 5.4K   | ✔    | 2.9MB | Coming soon                                                                |
| HearlthcareMagic | Multiple                  | EN       | ✔   | ×    | 200K   | ×    | 216MB | Coming soon                                                                |

https://github.com/Toyhom/Chinese-medical-dialogue-data

## 使用

### lora

- 显存 >= 13G （未量化版本）
- pip install deep_training cpm_kernels icetk transformers>=4.26.1
- torch >= 1.12.0 (icetk 依赖 cpu 版 torch, 建议先安装 icetk 后安装 gpu 版 torch)
- lora 的 finetune 代码来自 https://github.com/ssbuild/chatglm_finetuning

对于 fp16 模型，直接使用 Doctor_GLM/chat_lora.ipynb，由于官方更新了 chatglm 的权重，我们将老版权重放在了
[old_pretrain_model](https://pan.baidu.com/s/1vuoBbOQVPJPAcurEfVRn7A?pwd=ahwc)
可以下载后解压到 old_pretrain_model 目录

量化的模型我们打了个包，使用方便，但是效果目前来看很成问题：INT4 需要大约 6G 显存，INT8 需要大约 8G 显存，在 Doctor_GLM/chat_lora_quant.ipynb 下使用

```python
from load_quantization import load_int
tokenizer, model = load_int('DoctorGLM-6B-INT8-6merge-int8.pt',8)
response, history = model.chat(tokenizer,
                               "我爷爷高血压可以喝咖啡吗",
                               history=[],
                               max_length=2048)
print(response)
```

模型下载链接：
[INT4](https://pan.baidu.com/s/1nHQ1EQ2OBuWCyBZKBnBHYw?pwd=x6l4) [INT8](https://pan.baidu.com/s/1v2hWl1dPnh8xoJzxtpbugw?pwd=y4hu)
量化方法均为分层的线性量化。
目前量化模型的性能**仍有较大问题**，后期我们会对量化方法和模型进行更新

### p-tuningv2

官方提供了 p-tuningv2 的实现，新版本权重可以在 hugging face 上下载，也可以从我们的链接下载 [pretrain_model](https://pan.baidu.com/s/1WaG-NQeXVR7BNZs_zlUFmQ?pwd=h88g)  
p-tuningv2 的权重在
[ptuning_weight](https://pan.baidu.com/s/1Yf56egVGwI0XN2iOLcEGSQ?pwd=r4p0) ， 下载后解压到 ckpt/ptuningv2 目录下, 然后使用 Doctor_GLM/chat_ptuning_v2.ipynb，根据需要调整 quantization_bit 为 4 或 8

## 模型在线部署

为了方便部署并随时调整模型生成回答时的参数，我们提供了基于 `Gradio` 库的部署代码，路径为 `Doctor_GLM/gradio.ipynb`。运行之后，访问本机的 7860 或者代码声明的其他端口即可以运行 Demo，模型在生成回答时的参数可以由用户自由调控。若想让部署的模型可以被局域网之外的其他用户访问，需要将 sharing 设置为 `True`（默认为`False`）。部署之后运行效果如下所示：

<p align="center">
  <img src="imgs/gradio_demo.gif" width=1300px/>
  <br/>
</p>

## 最近更新

- <img src="https://img.shields.io/badge/Version-0.0.1--alpha-brightgreen"> (2023.4.3) 初版的权重，来自 LoRA SFT 1 epcoh
- <img src="https://img.shields.io/badge/Version-0.0.2--alpha-brightgreen"> (2023.4.13) LoRA-INT4/8 量化权重，以及我们实验发现 LoRA 一直会丢失对话能力，放弃该方式，转向 P-Tuning
- <img src="https://img.shields.io/badge/Version-0.0.3--alpha-brightgreen"> (2023.4.18) P-Tuning 多轮对话数据集训练的新权重和 arxiv

## 即将到来的更新

- [ ] <img src="https://img.shields.io/badge/Version-0.0.4--alpha-brightgreen"> (2023.4.21) 对话中加入参考文献，模型上传到 huggingface

第一次运行会下载 chatGLM-6B 权重, 如果已有 chatGLM-6B 权重可以将 data_utils.py 里的路径修改为自己的权重目录

## 结果示例

<p align="center">
  <img src="imgs/3_ret.png" width=1300px/>
  <br/>
</p>
我们随机跑了100个结果，在 ./results目录下，两份json文件分别为由ChatGLM, DoctorGLM得到的结果，目前存在大量复读机。

## 开发者群

<p align="left">
  <img src="imgs/11682312010_.pic.jpg" width=200px/>
</p>
DoctorGLM开发者群，如果你也对基于ChatGLM的应用开发感兴趣，欢迎加入我们的讨论组。

## 引用

```
@article{xiong2023doctorglm,
      title={DoctorGLM: Fine-tuning your Chinese Doctor is not a Herculean Task},
      author={Honglin Xiong and Sheng Wang and Yitao Zhu and Zihao Zhao and Yuxiao Liu and Linlin Huang and Qian Wang and Dinggang Shen},
}
```
