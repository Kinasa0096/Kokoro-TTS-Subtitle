import gradio as gr

def echo(x):
    return x

gr.Interface(fn=echo, inputs="text", outputs="text").launch(share=True)


# === ГЛАВНЫЙ ЗАПУСК ===
if __name__ == "__main__":
    demo1 = ui()
    demo2 = tutorial()
    demo = gr.TabbedInterface([demo1, demo2], ["Multilingual TTS", "VoicePack Explanation"], title="Kokoro TTS")
    demo.queue().launch(share=True)
