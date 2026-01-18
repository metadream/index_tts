import io
import torch
import torchaudio
from IndexTTS2 import IndexTTS2


def infer(audio_prompt1, text, audio_prompt2=None):
    tts2 = IndexTTS2()

    # 如果只用一个音频提示，直接调用 infer 并返回结果
    if audio_prompt2 is None:
        return tts2.infer(spk_audio_prompt=audio_prompt1, text=text, verbose=True)

    # 如果有两个音频提示，则需要分段处理每段文本 + 对应音频
    segments = split_text_by_speaker(text, audio_prompt1, audio_prompt2)
    segment_wavs = []

    for text, audio, _ in segments:
        wav, sampling_rate = tts2.infer(audio, text, verbose=True)
        segment_wavs.append(wav)

    merged_wav = torch.cat(segment_wavs, dim=-1)
    return merged_wav, sampling_rate


def wav_to_bytes(wav_tensor: torch.Tensor, sampling_rate: int) -> io.BytesIO:
    """
    将 torch.Tensor wav 转成 BytesIO WAV 流
    """
    buf = io.BytesIO()
    torchaudio.save(buf, wav_tensor, sampling_rate, format="wav")
    buf.seek(0)
    return buf


def wav_to_file(output_path, wav_tensor: torch.Tensor, sampling_rate: int, ):
    torchaudio.save(output_path, wav_tensor, sampling_rate)


def split_text_by_speaker(text, audio_1, audio_2):
    import re
    # 匹配文本中所有 [S1] 或 [S2] 的片段
    pattern = r'(\[s?S?1\]|\[s?S?2\])\s*([\s\S]*?)(?=\[s?S?[12]\]|$)'
    matches = re.findall(pattern, text)
    if not matches:
        raise ValueError("No speaker tags found in the text: [S1]... [S2]...")

    # 分别生成标签、文本、对应音频
    labels, contents = zip(*matches)
    audios = [audio_1 if l.lower() == '[s1]' else audio_2 for l in labels]

    # 返回列表，附带原始顺序索引
    return sorted(zip(contents, audios, range(len(contents))), key=lambda x: x[2])