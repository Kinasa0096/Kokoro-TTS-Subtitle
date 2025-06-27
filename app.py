# Initalize a pipeline
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
            gr.Warning(f"Make sure the input text is in {Language}",duration=10)
            gr.Warning(f"Fallback to English Language",duration=5)
            pipeline = KPipeline(lang_code="a")
            last_used_language = "a"

def get_voice_names(repo_id):
    return [os.path.splitext(file.replace("voices/", ""))[0] for file in list_repo_files(repo_id) if file.startswith("voices/")]

def create_audio_dir():
    root_dir = os.getcwd()
    audio_dir = os.path.join(root_dir, "kokoro_audio")
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        print(f"Created directory: {audio_dir}")
    else:
        print(f"Directory already exists: {audio_dir}")
    return audio_dir

def clean_text(text):
    replacements = {
        "–": " ",
        "-": " ",
        "**": " ",
        "*": " ",
        "#": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F]|'
        r'[\U0001F300-\U0001F5FF]|'
        r'[\U0001F680-\U0001F6FF]|'
        r'[\U0001F700-\U0001F77F]|'
        r'[\U0001F780-\U0001F7FF]|'
        r'[\U0001F800-\U0001F8FF]|'
        r'[\U0001F900-\U0001F9FF]|'
        r'[\U0001FA00-\U0001FA6F]|'
        r'[\U0001FA70-\U0001FAFF]|'
        r'[\U00002702-\U000027B0]|'
        r'[\U0001F1E0-\U0001F1FF]'
        r'', flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tts_file_name(text, language):
    global temp_folder
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    text = text.replace(" ", "_")
    language = language.replace(" ", "_").strip()
    truncated_text = text[:20] if len(text) > 20 else text if len(text) > 0 else language
    random_string = uuid.uuid4().hex[:8].upper()
    file_name = f"{temp_folder}/{truncated_text}_{random_string}.wav"
    return file_name

import numpy as np
import wave
from pydub import AudioSegment, silence

def strip_silence(audio: AudioSegment, silence_thresh=-45, chunk_size=10):
    not_silence = silence.detect_nonsilent(audio, min_silence_len=chunk_size, silence_thresh=silence_thresh)
    if not not_silence:
        return audio
    start_trim = not_silence[0][0]
    end_trim = not_silence[-1][1]
    return audio[start_trim:end_trim]

def generate_and_save_audio(text, Language="American English", voice="af_bella", speed=1, remove_silence=False, keep_silence_up_to=0.05):
    text = clean_text(text)
    update_pipeline(Language)
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
    temp_audio_segments = []
    temp_word_timestamps = []
    global_offset = 0.0

    for i, result in enumerate(generator):
        audio_np = result.audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        temp_file = f"temp_chunk_{uuid.uuid4().hex}.wav"
        with wave.open(temp_file, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_bytes)
        chunk_audio = AudioSegment.from_wav(temp_file)
        trimmed_audio = strip_silence(chunk_audio, silence_thresh=-45)
        not_silence = silence.detect_nonsilent(chunk_audio, min_silence_len=10, silence_thresh=-45)
        left_trim = not_silence[0][0] / 1000.0 if not_silence else 0
        temp_audio_segments.append(trimmed_audio)
        tokens = result.tokens
        for t in tokens:
            start = (t.start_ts or 0) - left_trim + global_offset
            end = (t.end_ts or 0) - left_trim + global_offset
            temp_word_timestamps.append({
                "word": t.text,
                "start": max(0.0, round(start, 3)),
                "end": max(0.0, round(end, 3)),
            })
        global_offset += len(trimmed_audio) / 1000.0
        os.remove(temp_file)

    final_audio = temp_audio_segments[0]
    for seg in temp_audio_segments[1:]:
        final_audio += seg
    save_path = tts_file_name(text, Language)
    final_audio.export(save_path, format="wav")
    return save_path, temp_word_timestamps

def adjust_timestamps(word_timestamps):
    return word_timestamps

import string

def write_word_srt(word_level_timestamps, output_file="word.srt", skip_punctuation=True):
    with open(output_file, "w", encoding="utf-8") as f:
        index = 1
        for entry in word_level_timestamps:
            word = entry["word"]
            if skip_punctuation and all(char in string.punctuation for char in word):
                continue
            start_time = entry["start"]
            end_time = entry["end"]
            def format_srt_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                sec = int(seconds % 60)
                millisec = int((seconds % 1) * 1000)
                return f"{hours:02}:{minutes:02}:{sec:02},{millisec:03}"
            start_srt = format_srt_time(start_time)
            end_srt = format_srt_time(end_time)
            f.write(f"{index}\n{start_srt} --> {end_srt}\n{word}\n\n")
            index += 1

def split_line_by_char_limit(text, max_chars=30):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) <= max_chars:
            current_line = (current_line + " " + word).strip()
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        if len(current_line.split()) == 1 and len(lines) > 0:
            lines[-1] += " " + current_line
        else:
            lines.append(current_line)
    return "\n".join(lines)

def write_sentence_srt(word_level_timestamps, output_file="subtitles.srt", max_words=8, min_pause=0.1):
    subtitles = []
    subtitle_words = []
    start_time = None
    remove_punctuation = ['"',"—"]
    for i, entry in enumerate(word_level_timestamps):
        word = entry["word"]
        word_start = entry["start"]
        word_end = entry["end"]
        if word in remove_punctuation:
            continue
        if word in string.punctuation:
            if subtitle_words:
                subtitle_words[-1] = (subtitle_words[-1][0] + word, subtitle_words[-1][1])
            continue
        if start_time is None:
            start_time = word_start
        if subtitle_words:
            last_word_end = subtitle_words[-1][1]
            pause_duration = word_start - last_word_end
        else:
            pause_duration = 0
        if (word.endswith(('.', '!', '?')) and len(subtitle_words) >= 5) or len(subtitle_words) >= max_words or pause_duration > min_pause:
            end_time = subtitle_words[-1][1]
            subtitle_text = " ".join(w[0] for w in subtitle_words)
            subtitles.append((start_time, end_time, subtitle_text))
            subtitle_words = [(word, word_end)]
            start_time = word_start
            continue
        subtitle_words.append((word, word_end))
    if subtitle_words:
        end_time = subtitle_words[-1][1]
        subtitle_text = " ".join(w[0] for w in subtitle_words)
        subtitles.append((start_time, end_time, subtitle_text))
    def format_srt_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        sec = int(seconds % 60)
        millisec = int((seconds % 1) * 1000)
        return f"{hours:02}:{minutes:02}:{sec:02},{millisec:03}"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(subtitles, start=1):
            text = split_line_by_char_limit(text, max_chars=30)
            f.write(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")

import json
import re

def fix_punctuation(text):
    text = re.sub(r'\s([.,?!])', r'\1', text)
    text = text.replace('" ', '"')
    text = text.replace(' "', '"')
    text = text.replace('" ', '"')
    track = 0
    result = []
    for index, char in enumerate(text):
        if char == '"':
            track += 1
            result.append(char)
            if track % 2 == 0:
                result.append(' ')
        else:
            result.append(char)
    text = ''.join(result)
    return text.strip()

def make_json(word_timestamps, json_file_name):
    data = {}
    temp = []
    inside_quote = False
    start_time = word_timestamps[0]['start']
    end_time = word_timestamps[0]['end']
    words_in_sentence = []
    sentence_id = 0
    for i, word_data in enumerate(word_timestamps):
        word = word_data['word']
        word_start = word_data['start']
        word_end = word_data['end']
        words_in_sentence.append({'word': word, 'start': word_start, 'end': word_end})
        end_time = word_end
        if word == '"':
            if inside_quote:
                temp[-1] += '"'
            else:
                temp.append('"')
            inside_quote = not inside_quote
        else:
            temp.append(word)
        if word.endswith(('.', '?', '!')) and not inside_quote:
            if i + 1 < len(word_timestamps):
                next_word = word_timestamps[i + 1]['word']
                if next_word[0].islower():
                    continue
            sentence = " ".join(temp)
            sentence = fix_punctuation(sentence)
            data[sentence_id] = {
                'text': sentence,
                'duration': end_time - start_time,
                'start': start_time,
                'end': end_time,
                'words': words_in_sentence
            }
            temp = []
            words_in_sentence = []
            start_time = word_data['start']
            sentence_id += 1
    if temp:
        sentence = " ".join(temp)
        sentence = fix_punctuation(sentence)
        data[sentence_id] = {
            'text': sentence,
            'duration': end_time - start_time,
            'start': start_time,
            'end': end_time,
            'words': words_in_sentence
        }
    with open(json_file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    return json_file_name

def modify_filename(save_path: str, prefix: str = ""):
    directory, filename = os.path.split(save_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{prefix}{name}{ext}"
    return os.path.join(directory, new_filename)
import shutil
def save_current_data():
    if os.path.exists("./last"):
        shutil.rmtree("./last")
    os.makedirs("./last",exist_ok=True)

def KOKORO_TTS_API(text, Language="American English",voice="af_bella", speed=1,translate_text=False,remove_silence=False,keep_silence_up_to=0.05):
    if translate_text:    
        text = bulk_translate(text, Language, chunk_size=500)
    save_path, word_level_timestamps = generate_and_save_audio(
        text=text, Language=Language, voice=voice, speed=speed, remove_silence=remove_silence, keep_silence_up_to=keep_silence_up_to
    )
    if Language in ["American English", "British English"]:
        word_level_srt = modify_filename(save_path.replace(".wav", ".srt"), prefix="word_level_")
        normal_srt = modify_filename(save_path.replace(".wav", ".srt"), prefix="sentence_")
        json_file = modify_filename(save_path.replace(".wav", ".json"), prefix="duration_")
        write_word_srt(word_level_timestamps, output_file=word_level_srt, skip_punctuation=True)
        write_sentence_srt(word_level_timestamps, output_file=normal_srt, min_pause=0.01)
        make_json(word_level_timestamps, json_file)
        save_current_data()
        shutil.copy(save_path, "./last/")
        shutil.copy(word_level_srt, "./last/")
        shutil.copy(normal_srt, "./last/")
        shutil.copy(json_file, "./last/")
        return save_path, save_path, word_level_srt, normal_srt, json_file
    return save_path, save_path, None, None, None    

def ui():
    def toggle_autoplay(autoplay):
        return gr.Audio(interactive=False, label='Output Audio', autoplay=autoplay)
    dummy_examples = [
        ["Hey, y'all, let’s grab some coffee and catch up!", "American English", "af_bella"],
        ["I'd like a large coffee, please.", "British English", "bf_isabella"],
        ["नमस्ते, कैसे हो?", "Hindi", "hf_alpha"],
        ["Hola, ¿cómo estás?", "Spanish", "ef_dora"],
        ["Bonjour, comment ça va?", "French", "ff_siwis"],
        ["Ciao, come stai?", "Italian", "if_sara"],
        ["Olá, como você está?", "Brazilian Portuguese", "pf_dora"],
        ["こんにちは、お元気ですか？", "Japanese", "jf_nezumi"],
        ["你好，你怎么样?", "Mandarin Chinese", "zf_xiaoni"]
    ]
    with gr.Blocks() as demo:
        lang_list = ['American English', 'British English', 'Hindi', 'Spanish', 'French', 'Italian', 'Brazilian Portuguese', 'Japanese', 'Mandarin Chinese']
        voice_names = get_voice_names("hexgrad/Kokoro-82M")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label='📝 Enter Text', lines=3)
                with gr.Row():
                    language_name = gr.Dropdown(lang_list, label="🌍 Select Language", value=lang_list[0])
                with gr.Row():
                    voice_name = gr.Dropdown(voice_names, label="🎙️ Choose VoicePack", value='af_heart')
                with gr.Row():
                    generate_btn = gr.Button('🚀 Generate', variant='primary')
                with gr.Accordion('🎛️ Audio Settings', open=False):
                    speed = gr.Slider(minimum=0.25, maximum=2, value=1, step=0.1, label='⚡️Speed', info='Adjust the speaking speed')
                    translate_text = gr.Checkbox(value=False, label='🌐 Translate Text to Selected Language')
                    remove_silence = gr.Checkbox(value=False, label='✂️ Remove Silence From TTS')
            with gr.Column():
                audio = gr.Audio(interactive=False, label='🔊 Output Audio', autoplay=True)
                audio_file = gr.File(label='📥 Download Audio')
                with gr.Accordion('🎬 Autoplay, Subtitle, Timestamp', open=False):
                    autoplay = gr.Checkbox(value=True, label='▶️ Autoplay')
                    autoplay.change(toggle_autoplay, inputs=[autoplay], outputs=[audio])
                    word_level
