# app.py — ПОЛНОСТЬЮ ГОТОВЫЙ ФАЙЛ, исправляет длинные паузы между чанками + сабы по предложениям

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
        pipeline = KPipeline(lang_code=new_lang)
        last_used_language = new_lang
        try:
            pipeline = KPipeline(lang_code=new_lang)
            last_used_language = new_lang
        except Exception as e:
            gr.Warning(f"Make sure the input text is in {Language}", duration=10)
            gr.Warning(f"Fallback to English Language", duration=5)
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
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F700-\U0001F77F]|[\U0001F780-\U0001F7FF]|[\U0001F800-\U0001F8FF]|[\U0001F900-\U0001F9FF]|[\U0001FA00-\U0001FA6F]|[\U0001FA70-\U0001FAFF]|[\U00002702-\U000027B0]|[\U0001F1E0-\U0001F1FF]', flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tts_file_name(text, language):
    global temp_folder
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip().replace(" ", "_")
    language = language.replace(" ", "_").strip()
    truncated_text = text[:20] if len(text) > 20 else text if len(text) > 0 else language
    random_string = uuid.uuid4().hex[:8].upper()
    return f"{temp_folder}/{truncated_text}_{random_string}.wav"

import numpy as np
import wave

def trim_trailing_silence(audio_np, sample_rate=24000, threshold=0.001, min_silence_sec=0.05):
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()
    window_size = int(sample_rate * min_silence_sec)
    for i in range(len(audio_np) - 1, window_size, -1):
        if np.max(np.abs(audio_np[i-window_size:i])) > threshold:
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
    last_end = 0
    for i in sorted(timestamp_dict):
        seg = timestamp_dict[i]
        words = seg["words"]
        dur = seg["duration"]
        last_word_end = max((w["end"] for w in words if w["end"]), default=0)
        for w in words:
            start = w["start"] or 0
            end = w["end"] or start
            adjusted.append({"word": w["word"], "start": round(last_end + start, 3), "end": round(last_end + end, 3)})
        last_end += dur
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
    sentences = [s.strip() for s in text.strip().split('\n') if s.strip()]
    i = 0
    subs = []
    for sent in sentences:
        words = sent.strip().split()
        matched = []
        while i < len(word_timestamps) and len(matched) < len(words):
            w1 = word_timestamps[i]['word'].strip(string.punctuation).lower()
            w2 = words[len(matched)].strip(string.punctuation).lower()
            if w1 == w2: matched.append(word_timestamps[i])
            i += 1
        if matched:
            start = matched[0]['start']
            end = matched[-1]['end']
            subs.append((start, end, sent))
    def fmt(s): return f"{int(s//3600):02}:{int((s%3600)//60):02}:{int(s%60):02},{int((s%1)*1000):03}"
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, (s, e, t) in enumerate(subs, 1):
            f.write(f"{idx}\n{fmt(s)} --> {fmt(e)}\n{t}\n\n")

def make_json(word_timestamps, json_file_name):
    import json
    data = {}
    i = 0
    start = word_timestamps[0]['start']
    words = []
    for w in word_timestamps:
        words.append({'word': w['word'], 'start': w['start'], 'end': w['end']})
    data[i] = {'text': ' '.join(w['word'] for w in words), 'start': start, 'end': words[-1]['end'], 'duration': words[-1]['end'] - start, 'words': words}
    with open(json_file_name, 'w') as f:
        json.dump(data, f, indent=4)
    return json_file_name

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
        json_f = modify_filename(save_path.replace(".wav", ".json"), "duration_")
        write_word_srt(wts, word_srt)
        write_sentence_srt_from_text(text, wts, sent_srt)
        make_json(wts, json_f)
        save_current_data()
        for f in [save_path, word_srt, sent_srt, json_f]:
            shutil.copy(f, "./last/")
        return save_path, save_path, word_srt, sent_srt, json_f
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
                    js_file = gr.File(label='⏳ Sentence Timestamp JSON')
        text.submit(KOKORO_TTS_API, [text, language, voice, speed, trans, rm_sil], [audio, a_file, word_file, sent_file, js_file])
        btn.click(KOKORO_TTS_API, [text, language, voice, speed, trans, rm_sil], [audio, a_file, word_file, sent_file, js_file])
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
