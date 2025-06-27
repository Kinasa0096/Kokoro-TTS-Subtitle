# app.py — исправленная версия, в которой sentence-level сабы строго 1 предложение = 1 субтитр

from kokoro import KPipeline
import os
from huggingface_hub import list_repo_files
import uuid
import re 
import gradio as gr
from deep_translator import GoogleTranslator

def bulk_translate(text, target_language, chunk_size=500):
    language_map_local = {
        "American English": "en",
        "British English": "en",
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "Italian": "it",
        "Brazilian Portuguese": "pt",
        "Japanese": "ja",
        "Mandarin Chinese": "zh-CN"
    }
    lang_code = language_map_local[target_language]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    translated_chunks = [GoogleTranslator(target=lang_code).translate(chunk) for chunk in chunks]
    result = " ".join(translated_chunks)
    return result.strip()

language_map = {
    "American English": "a",
    "British English": "b",
    "Hindi": "h",
    "Spanish": "e",
    "French": "f",
    "Italian": "i",
    "Brazilian Portuguese": "p",
    "Japanese": "j",
    "Mandarin Chinese": "z"
}

def update_pipeline(Language):
    global pipeline, last_used_language
    new_lang = language_map.get(Language, "a")
    if new_lang != last_used_language:
        try:
            pipeline = KPipeline(lang_code=new_lang)
            last_used_language = new_lang
        except Exception:
            pipeline = KPipeline(lang_code="a")
            last_used_language = "a"

def get_voice_names(repo_id):
    return [os.path.splitext(file.replace("voices/", ""))[0] for file in list_repo_files(repo_id) if file.startswith("voices/")]

def create_audio_dir():
    root_dir = os.getcwd()
    audio_dir = os.path.join(root_dir, "kokoro_audio")
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    return audio_dir

def clean_text(text):
    replacements = {"–": " ", "-": " ", "**": " ", "*": " ", "#": " "}
    for old, new in replacements.items():
        text = text.replace(old, new)
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\u2702-\u27B0\U0001F1E0-\U0001F1FF]')
    text = emoji_pattern.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tts_file_name(text, language):
    global temp_folder
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip().replace(" ", "_")
    language = language.replace(" ", "_").strip()
    truncated = text[:20] if len(text) > 20 else text or language
    random_str = uuid.uuid4().hex[:8].upper()
    return f"{temp_folder}/{truncated}_{random_str}.wav"

import numpy as np
import wave

def trim_trailing_silence(audio_np, sample_rate=24000, threshold=0.001, min_silence_sec=0.05):
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()
    win = max(1, int(sample_rate * min_silence_sec))
    for i in range(len(audio_np) - 1, win, -1):
        if np.max(np.abs(audio_np[i - win:i])) > threshold:
            return audio_np[:i]
    return audio_np

def generate_and_save_audio(text, Language="American English", voice="af_bella", speed=1, remove_silence=False, keep_silence_up_to=0.01):
    text = clean_text(text)
    update_pipeline(Language)
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
    save_path = tts_file_name(text, Language)
    timestamps = {}
    with wave.open(save_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        for i, result in enumerate(generator):
            audio_np = result.audio.numpy()
            audio_np = trim_trailing_silence(audio_np)
            audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
            wav_file.writeframes(audio_bytes)
            tokens = result.tokens
            timestamps[i] = {
                "text": result.graphemes,
                "words": [{"word": t.text, "start": t.start_ts, "end": t.end_ts} for t in tokens],
                "duration": len(audio_np) / 24000
            }
    return save_path, timestamps

def adjust_timestamps(timestamp_dict):
    adjusted = []
    offset = 0
    for i in sorted(timestamp_dict):
        chunk = timestamp_dict[i]
        for w in chunk["words"]:
            start = round(offset + (w["start"] or 0), 3)
            end = round(offset + (w["end"] or start), 3)
            adjusted.append({"word": w["word"], "start": start, "end": end})
        offset += chunk["duration"]
    return adjusted

def write_word_srt(timestamps, output_file="word.srt"):
    import string
    def fmt(s): return f"{int(s//3600):02}:{int((s%3600)//60):02}:{int(s%60):02},{int((s%1)*1000):03}"
    with open(output_file, "w", encoding="utf-8") as f:
        idx = 1
        for w in timestamps:
            if all(c in string.punctuation for c in w["word"]): continue
            f.write(f"{idx}\n{fmt(w['start'])} --> {fmt(w['end'])}\n{w['word']}\n\n")
            idx += 1

def write_sentence_srt_from_text(text, word_timestamps, output_file="subtitles.srt"):
    import string
    sentence_list = [s.strip() for s in text.strip().split('\n') if s.strip()]
    subtitles = []
    word_idx = 0

    for sentence in sentence_list:
        sentence_words = sentence.strip().split()
        matched = []
        for sw in sentence_words:
            sw_clean = sw.strip(string.punctuation).lower()
            while word_idx < len(word_timestamps):
                wt = word_timestamps[word_idx]
                w_clean = wt["word"].strip(string.punctuation).lower()
                word_idx += 1
                if sw_clean == w_clean:
                    matched.append(wt)
                    break
        if matched:
            start_time = matched[0]['start']
            end_time = matched[-1]['end']
            subtitles.append((start_time, end_time, sentence))

    def fmt(s): return f"{int(s//3600):02}:{int((s%3600)//60):02}:{int(s%60):02},{int((s%1)*1000):03}"
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(subtitles, 1):
            f.write(f"{idx}\n{fmt(start)} --> {fmt(end)}\n{text}\n\n")

import shutil
def save_current_data():
    if os.path.exists("./last"):
        shutil.rmtree("./last")
    os.makedirs("./last", exist_ok=True)

def modify_filename(save_path, prefix=""):
    d, f = os.path.split(save_path)
    n, e = os.path.splitext(f)
    return os.path.join(d, f"{prefix}{n}{e}")

def KOKORO_TTS_API(text, Language="American English", voice="af_bella", speed=1, translate_text=False, remove_silence=False, keep_silence_up_to=0.05):
    if translate_text:
        text = bulk_translate(text, Language)
    save_path, ts = generate_and_save_audio(text, Language, voice, speed, remove_silence, keep_silence_up_to)
    if not remove_silence and Language in ["American English", "British English"]:
        wts = adjust_timestamps(ts)
        word_srt = modify_filename(save_path.replace(".wav", ".srt"), "word_level_")
        sent_srt = modify_filename(save_path.replace(".wav", ".srt"), "sentence_")
        write_word_srt(wts, word_srt)
        write_sentence_srt_from_text(text, wts, sent_srt)
        save_current_data()
        for f in [save_path, word_srt, sent_srt]:
            shutil.copy(f, "./last/")
        return save_path, save_path, word_srt, sent_srt, None
    return save_path, save_path, None, None, None

def ui():
    def toggle_autoplay(autoplay):
        return gr.Audio(interactive=False, label='Output Audio', autoplay=autoplay)
    langs = ['American English', 'British English', 'Hindi', 'Spanish', 'French', 'Italian', 'Brazilian Portuguese', 'Japanese', 'Mandarin Chinese']
    voices = get_voice_names("hexgrad/Kokoro-82M")
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label='📝 Enter Text', lines=3)
                language = gr.Dropdown(langs, label="🌍 Select Language", value=langs[0])
                voice = gr.Dropdown(voices, label="🎙️ Choose VoicePack", value='af_heart')
                btn = gr.Button('🚀 Generate', variant='primary')
                with gr.Accordion('🎛️ Audio Settings', open=False):
                    speed = gr.Slider(0.25, 2, 1, 0.1, label='⚡️Speed')
                    trans = gr.Checkbox(False, label='🌐 Translate Text')
                    rm_sil = gr.Checkbox(False, label='✂️ Remove Silence')
            with gr.Column():
                audio = gr.Audio(interactive=False, label='🔊 Output Audio', autoplay=True)
                a_file = gr.File(label='📥 Download Audio')
                with gr.Accordion('🎬 Subtitle & Timestamp', open=False):
                    auto = gr.Checkbox(True, label='▶️ Autoplay')
                    auto.change(toggle_autoplay, inputs=[auto], outputs=[audio])
                    word_file = gr.File(label='📝 Word-Level SRT')
                    sent_file = gr.File(label='📜 Sentence-Level SRT')
        text.submit(KOKORO_TTS_API, [text, language, voice, speed, trans, rm_sil], [audio, a_file, word_file, sent_file, gr.File()])
        btn.click(KOKORO_TTS_API, [text, language, voice, speed, trans, rm_sil], [audio, a_file, word_file, sent_file, gr.File()])
    return demo

import click
@click.command()
@click.option("--debug", is_flag=True, default=False)
@click.option("--share", is_flag=True, default=False)
def main(debug, share):
    demo = ui()
    gr.TabbedInterface([demo], ["Multilingual TTS"], title="Kokoro TTS").queue().launch(debug=debug, share=share)

last_used_language = "a"
pipeline = KPipeline(lang_code=last_used_language)
temp_folder = create_audio_dir()

if __name__ == "__main__":
    main()
