# IndexTTS2 部署指南

本项目在 https://github.com/index-tts/index-tts 基础上进行简化，仅提供原始推理方法。

## 安装系统依赖
```bash
apt update
apt install -y python3 python3-pip python3-venv
```

## 克隆仓库
```bash
git clone https://github.com/metadream/index_tts
cd index_tts
```

## 创建虚拟环境
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 安装应用依赖
```bash
pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
# pip3 install -r requirements.txt
```

如果在GPU环境下运行，还可以安装deepspeed和cuda加速模块：
```bash
pip install deepspeed==0.17.1
pip install torch==2.8.* torchaudio==2.8.* torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

## 下载模型

与官方仓库运行时自动下载模型不同，本项目需要预先手动下载所有模型，并统一存放在 `checkpoints`
目录下，因此本章节所有命令均在 `checkpoints` 目录下执行。

```bash
# 1. 下载基础模型，然后删除其中的.git目录避免冲突
git clone --depth 1 https://huggingface.co/IndexTeam/IndexTTS-2 ./
rm -rf .git

# 2. 下载说话人识别模型
wget https://hf-mirror.com/funasr/campplus/resolve/main/campplus_cn_common.bin

# 3. 下载语义编码器
wget -P semantic_codec https://hf-mirror.com/amphion/MaskGCT/resolve/main/semantic_codec/model.safetensors

# 4. 下载自监督语音预训练模型，并删除其中的.git目录
git clone --depth 1 https://huggingface.co/facebook/w2v-bert-2.0 ./w2v-bert-2.0
rm -rf w2v-bert-2.0/.git

# 5. 下载非自回归语音合成模型，并删除其中的.git目录
git clone --depth 1 https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x ./bigvgan/bigvgan_v2_22khz_80band_256x
rm -rf bigvgan/bigvgan_v2_22khz_80band_256x/.git
```

## 调用方法
```python
from inference import IndexTTS2
tts = IndexTTS2(use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "开心一刻也是地久天长!"
tts.infer(spk_audio_prompt='voice_01.wav', text=text, output_path="output.wav", verbose=True)
```