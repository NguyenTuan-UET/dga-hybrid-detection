"""
Web demo cho Bilbo DGA Detector — giao diện sáng, tương phản cao, phẳng (không gradient).

Chạy:
    python app.py
Rồi mở http://127.0.0.1:7860
"""
import os
import re
import sys
from html import escape

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # bớt log TensorFlow

import gradio as gr

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'src'))

from predict import load_models, predict, DEFAULT_MODEL_DIR, THRESHOLD

print("Đang nạp model...")
MODELS = load_models(DEFAULT_MODEL_DIR)
print("Sẵn sàng!")


# ── Bảng màu (sáng, tương phản cao, phẳng) ─────────────────────────────────
RED        = "#EF2D3C"   # đỏ thương hiệu / nút chính
RED_DK     = "#D81F2D"   # đỏ đậm khi hover
INK        = "#15161A"   # chữ tiêu đề (gần đen)
BODY       = "#3A3B40"   # chữ thường
MUTED      = "#8A8B92"   # chữ phụ
BORDER     = "#E7E7EA"   # viền
PANEL      = "#FFFFFF"   # nền thẻ
PAGE       = "#FFFFFF"   # nền trang
SAFE       = "#15A34A"   # xanh lành tính
SAFE_BG    = "#E9F8EF"
DANGER     = "#E5141F"   # đỏ độc hại
DANGER_BG  = "#FDEAEB"


EMPTY_HTML = """
<div class="empty-state">
  <div class="empty-icon">🔍</div>
  <p>Nhập domain bên trái rồi bấm <b>Kiểm tra</b><br>để xem kết quả phân tích.</p>
</div>
"""


def render_results(domains, probs, threshold):
    if not domains:
        return EMPTY_HTML

    n_dga = sum(1 for p in probs if p >= threshold)
    parts = [
        f'<div class="summary"><span class="s-total">{len(domains)} domain</span>'
        f'<span class="s-dot">·</span>'
        f'<span class="s-dga">{n_dga} nghi ngờ DGA</span>'
        f'<span class="s-dot">·</span>'
        f'<span class="s-safe">{len(domains) - n_dga} lành tính</span></div>'
    ]

    for d, p in zip(domains, probs):
        p = float(p)
        is_dga = p >= threshold
        cls   = "danger" if is_dga else "safe"
        label = "DGA — độc hại" if is_dga else "Lành tính"
        icon  = "⚠️" if is_dga else "✅"
        conf  = p * 100 if is_dga else (1 - p) * 100
        parts.append(f"""
        <div class="card {cls}">
          <div class="card-top">
            <span class="domain">{escape(d)}</span>
            <span class="badge {cls}">{icon} {label}</span>
          </div>
          <div class="bar"><div class="fill {cls}" style="width:{p*100:.1f}%"></div></div>
          <div class="meta">Điểm DGA <b>{p:.4f}</b><span class="dot">·</span>Độ tin cậy {conf:.1f}%</div>
        </div>""")
    return "".join(parts)


def check_domains(text, threshold):
    domains = [d for d in re.split(r'[\s,;]+', text.strip()) if d]
    if not domains:
        return EMPTY_HTML
    cleaned, probs = predict(MODELS, domains)
    return render_results(cleaned, probs, threshold)


# ── CSS ────────────────────────────────────────────────────────────────────
CSS = f"""
.gradio-container {{
    background: {PAGE} !important;
    max-width: 1080px !important;
    margin: 0 auto !important;
    font-family: "Inter", ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
footer {{ display: none !important; }}

#header {{ text-align: center; padding: 36px 0 14px; }}
#header .logo-box {{
    width: 56px; height: 56px; margin: 0 auto 14px;
    background: {RED}; border-radius: 15px;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; line-height: 1;
}}
#header h1 {{
    font-size: 34px; font-weight: 800; color: {INK};
    margin: 0 0 6px; letter-spacing: -0.6px;
}}
#header h1 .accent {{ color: {RED}; }}
#header p {{ color: {MUTED}; font-size: 15px; margin: 0; }}

/* thẻ panel chung — phẳng, viền rõ */
.panel {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 1px 2px rgba(20,20,30,0.03);
}}
.panel-label {{
    font-weight: 700; font-size: 14px; color: {INK};
    margin-bottom: 12px; display: block;
}}

/* nút chính — đỏ đặc, không gradient */
#check-btn {{
    background: {RED} !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    border-radius: 11px !important;
    box-shadow: none !important;
    transition: background .15s ease;
}}
#check-btn:hover {{ background: {RED_DK} !important; }}

/* ô nhập */
#dom-input textarea {{
    border-radius: 11px !important;
    border: 1px solid {BORDER} !important;
    background: #FFFFFF !important;
    font-size: 14.5px !important;
    color: {INK} !important;
}}
#dom-input textarea:focus {{
    border-color: {RED} !important;
    box-shadow: 0 0 0 3px rgba(239,45,60,0.12) !important;
}}

/* khu kết quả */
.results {{ min-height: 320px; }}
.summary {{
    font-size: 13px; color: {MUTED};
    padding: 2px 2px 14px; display: flex; gap: 8px; align-items: center;
    border-bottom: 1px solid {BORDER}; margin-bottom: 14px;
}}
.summary .s-total {{ color: {INK}; font-weight: 700; }}
.summary .s-dga  {{ color: {DANGER}; font-weight: 700; }}
.summary .s-safe {{ color: {SAFE}; font-weight: 700; }}
.summary .s-dot  {{ color: #D5D5DA; }}

.card {{
    background: #fff;
    border: 1px solid {BORDER};
    border-radius: 13px;
    padding: 14px 16px;
    margin-bottom: 11px;
}}
.card.danger {{ border-left: 4px solid {DANGER}; }}
.card.safe   {{ border-left: 4px solid {SAFE}; }}

.card-top {{ display:flex; justify-content:space-between; align-items:center; gap:12px; }}
.card .domain {{ font-weight:700; font-size:15px; color:{INK}; word-break:break-all; }}
.card .badge {{
    font-size:12px; font-weight:700; padding:4px 11px;
    border-radius:999px; white-space:nowrap;
}}
.badge.danger {{ background:{DANGER_BG}; color:{DANGER}; }}
.badge.safe   {{ background:{SAFE_BG}; color:{SAFE}; }}

/* thanh điểm — màu đặc, KHÔNG gradient */
.card .bar {{
    height:7px; background:#EFEFF2; border-radius:999px;
    margin:12px 0 9px; overflow:hidden;
}}
.card .fill {{ height:100%; border-radius:999px; }}
.fill.danger {{ background:{DANGER}; }}
.fill.safe   {{ background:{SAFE}; }}
.card .meta {{ font-size:12.5px; color:{MUTED}; }}
.card .meta b {{ color:{BODY}; }}
.card .meta .dot {{ margin:0 7px; color:#D5D5DA; }}

.empty-state {{ text-align:center; color:{MUTED}; padding:70px 20px; }}
.empty-state .empty-icon {{ font-size:36px; opacity:.45; margin-bottom:10px; }}
.empty-state p {{ font-size:14.5px; line-height:1.6; }}
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.gray,
).set(
    body_background_fill=PAGE,
    block_background_fill=PANEL,
)


with gr.Blocks(title="Bilbo DGA Detector") as demo:
    gr.HTML(
        """
        <div id="header">
          <div class="logo-box">🛡️</div>
          <h1>Bilbo <span class="accent">DGA Detector</span></h1>
          <p>Phát hiện domain Dictionary DGA bằng mô hình deep learning (CNN + LSTM)</p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            with gr.Group(elem_classes="panel"):
                gr.HTML('<span class="panel-label">Domain cần kiểm tra</span>')
                inp = gr.Textbox(
                    show_label=False,
                    placeholder="google.com\nmortiscontrastatim.com\njeannetteabrahamson.net",
                    lines=8,
                    elem_id="dom-input",
                )
                threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=THRESHOLD, step=0.05,
                    label="Ngưỡng phân loại (hạ thấp để bắt nhiều DGA lạ hơn)",
                )
                btn = gr.Button("Kiểm tra", variant="primary", elem_id="check-btn")

            gr.Examples(
                examples=[
                    ["google.com\npaypal.com\ngithub.com", 0.5],
                    ["mortiscontrastatim.com\njeannetteabrahamson.net\nthinkhomereach.art", 0.5],
                    ["txumyqrubwutbb.cc\nbelievekeylab.in", 0.4],
                ],
                inputs=[inp, threshold],
                label="Ví dụ — bấm để điền nhanh",
            )

        with gr.Column(scale=6):
            with gr.Group(elem_classes="panel"):
                gr.HTML('<span class="panel-label">Kết quả phân tích</span>')
                out = gr.HTML(EMPTY_HTML, elem_classes="results")

    btn.click(check_domains, inputs=[inp, threshold], outputs=out)
    inp.submit(check_domains, inputs=[inp, threshold], outputs=out)


if __name__ == '__main__':
    demo.launch(theme=theme, css=CSS)
