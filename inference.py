import io
import os

import torch
import torchaudio
from IndexTTS2 import IndexTTS2


# =========================================================
# IndexTTS2 接口适配与扩展：支持双人语音对话模式
# =========================================================

def infer(text_prompt, audio_prompt1, audio_prompt2=None, verbose=False):
    tts2 = IndexTTS2()

    # 如果只输入一个音频提示，直接返回推理结果
    if audio_prompt2 is None:
        return tts2.infer(
            text=text_prompt,
            spk_audio_prompt=audio_prompt1,
            verbose=verbose
        )

    # 如果输入两个音频提示，则分角色处理每段文本和对应音频
    segments = split_text_by_speaker(text_prompt, audio_prompt1, audio_prompt2)
    segment_wavs = []

    for text, audio, _ in segments:
        wav, sampling_rate = tts2.infer(text, audio, verbose=verbose)
        segment_wavs.append(wav)
    # 拼接音频片段
    merged_wav = torch.cat(segment_wavs, dim=-1)
    return merged_wav, sampling_rate


# 根据说话人标签分割文本和音频
def split_text_by_speaker(text, audio_1, audio_2):
    import re
    # 匹配文本中所有 [S1] 或 [S2] 片段
    pattern = r'(\[s?S?1\]|\[s?S?2\])\s*([\s\S]*?)(?=\[s?S?[12]\]|$)'
    matches = re.findall(pattern, text)
    if not matches:
        raise ValueError("No speaker tags found in the text: [S1]... [S2]...")

    # 分别生成标签、文本、对应音频
    labels, contents = zip(*matches)
    audios = [audio_1 if l.lower() == '[s1]' else audio_2 for l in labels]
    # 返回列表，附带原始顺序索引
    return sorted(zip(contents, audios, range(len(contents))), key=lambda x: x[2])


# 将 torch.Tensor wav 转成字节流
def wav_to_bytes(wav_tensor: torch.Tensor, sampling_rate: int) -> io.BytesIO:
    buff = io.BytesIO()
    torchaudio.save(buff, wav_tensor, sampling_rate, format="wav")
    buff.seek(0)
    return buff


# 将 torch.Tensor wav 保存到文件
def wav_to_file(output_path: str, wav_tensor: torch.Tensor, sampling_rate: int) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, wav_tensor, sampling_rate)
    return output_path