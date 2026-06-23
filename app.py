"""
Web demo cho Bilbo DGA Detector.

Chạy:
    python app.py

Rồi mở trình duyệt tại http://127.0.0.1:7860
Gõ domain (mỗi dòng 1 cái), bấm "Kiểm tra" để xem domain nào là DGA.
"""
import os
import re
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # bớt log TensorFlow

import gradio as gr

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'src'))

from predict import load_models, predict, DEFAULT_MODEL_DIR, THRESHOLD

# Nạp model 1 lần lúc khởi động (ensemble bilbo_run*.keras)
print("Đang nạp model...")
MODELS = load_models(DEFAULT_MODEL_DIR)
print("Sẵn sàng!")


def check_domains(text, threshold):
    """Nhận chuỗi nhiều domain (cách nhau bởi xuống dòng/dấu phẩy/khoảng trắng)."""
    domains = [d for d in re.split(r'[\s,;]+', text.strip()) if d]
    if not domains:
        return [["—", "—", "Hãy nhập ít nhất 1 domain"]]

    cleaned, probs = predict(MODELS, domains)

    rows = []
    for d, p in zip(cleaned, probs):
        if p >= threshold:
            verdict = f"⚠️ DGA (độc hại) — tin cậy {p*100:.1f}%"
        else:
            verdict = f"✅ Lành tính — tin cậy {(1-p)*100:.1f}%"
        rows.append([d, f"{p:.4f}", verdict])
    return rows


with gr.Blocks(title="Bilbo DGA Detector") as demo:
    gr.Markdown(
        """
        # 🛡️ Bilbo — Phát hiện domain DGA

        Nhập một hoặc nhiều domain (mỗi dòng 1 cái). Model sẽ chấm điểm khả năng
        domain được sinh ra bởi **Dictionary DGA** (kỹ thuật malware tạo domain
        để liên lạc với máy chủ điều khiển).

        - **Score gần 0** → lành tính
        - **Score gần 1** → khả nghi (DGA)
        """
    )

    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(
                label="Domain cần kiểm tra",
                placeholder="google.com\nmortiscontrastatim.com\nbrothernerveplacebringconsult.com",
                lines=6,
            )
            threshold = gr.Slider(
                minimum=0.0, maximum=1.0, value=THRESHOLD, step=0.05,
                label="Ngưỡng phân loại (hạ thấp để bắt được nhiều DGA lạ hơn)",
            )
            btn = gr.Button("Kiểm tra", variant="primary")
        with gr.Column():
            out = gr.Dataframe(
                headers=["Domain", "Score", "Kết luận"],
                label="Kết quả",
                wrap=True,
            )

    gr.Examples(
        examples=[
            ["google.com\npaypal.com\ngithub.com", 0.5],
            ["mortiscontrastatim.com\njourneyvalley.net\nthinkhomereach.art", 0.5],
        ],
        inputs=[inp, threshold],
    )

    btn.click(check_domains, inputs=[inp, threshold], outputs=out)
    inp.submit(check_domains, inputs=[inp, threshold], outputs=out)


if __name__ == '__main__':
    demo.launch()
