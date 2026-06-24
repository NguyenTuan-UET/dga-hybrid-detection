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
PAGE       = "#F4F5F7"   # nền trang (xám rất nhạt để thẻ trắng nổi lên)
SUBTLE     = "#FAFAFB"   # nền phụ bên trong thẻ
SAFE       = "#15A34A"   # xanh lành tính
SAFE_BG    = "#E9F8EF"
DANGER     = "#E5141F"   # đỏ độc hại
DANGER_BG  = "#FDEAEB"


EMPTY_HTML = """
<div class="empty-state">
  <div class="empty-icon">🔍</div>
  <p>Nhập domain ở cột bên trái rồi bấm <b>Kiểm tra</b><br>để xem kết quả phân tích.</p>
</div>
"""


def render_results(domains, probs, threshold):
    if not domains:
        return EMPTY_HTML

    n_dga = sum(1 for p in probs if p >= threshold)
    n_safe = len(domains) - n_dga
    parts = [
        '<div class="summary">'
        f'<div class="stat"><span class="stat-num">{len(domains)}</span>'
        f'<span class="stat-lbl">Tổng domain</span></div>'
        f'<div class="stat danger"><span class="stat-num">{n_dga}</span>'
        f'<span class="stat-lbl">Nghi ngờ DGA</span></div>'
        f'<div class="stat safe"><span class="stat-num">{n_safe}</span>'
        f'<span class="stat-lbl">Lành tính</span></div>'
        '</div>'
        '<div class="cards">'
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
          <div class="meta"><span>Điểm DGA <b>{p:.4f}</b></span>
            <span class="dot">·</span><span>Độ tin cậy <b>{conf:.1f}%</b></span></div>
        </div>""")
    parts.append('</div>')
    return "".join(parts)


def check_domains(text, threshold):
    domains = [d for d in re.split(r'[\s,;]+', text.strip()) if d]
    if not domains:
        return EMPTY_HTML
    cleaned, probs = predict(MODELS, domains)
    return render_results(cleaned, probs, threshold)


# ── Ép giao diện luôn chạy ở chế độ sáng (light) ───────────────────────────
# Chạy ngay trong <head> trước khi Gradio vẽ trang -> không bị nháy nền tối,
# bất kể trình duyệt/hệ điều hành đang để dark mode.
FORCE_LIGHT_HEAD = """
<script>
(function () {
  try {
    var u = new URL(window.location.href);
    if (u.searchParams.get('__theme') !== 'light') {
      u.searchParams.set('__theme', 'light');
      window.location.replace(u.toString());
    }
  } catch (e) {}
})();
</script>
"""


# ── CSS ────────────────────────────────────────────────────────────────────
CSS = f"""
.gradio-container {{
    background: {PAGE} !important;
    max-width: 1120px !important;
    margin: 0 auto !important;
    padding: 0 16px 40px !important;
    font-family: "Inter", ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
footer {{ display: none !important; }}
.gradio-container .gap {{ gap: 18px !important; }}

/* ── Header ──────────────────────────────────────────────────────────── */
#header {{ text-align: center; padding: 40px 0 22px; }}
#header .logo-box {{
    width: 60px; height: 60px; margin: 0 auto 16px;
    background: {RED}; border-radius: 17px;
    display: flex; align-items: center; justify-content: center;
    font-size: 30px; line-height: 1;
    box-shadow: 0 8px 22px rgba(239,45,60,0.28);
}}
#header h1 {{
    font-size: 36px; font-weight: 800; color: {INK};
    margin: 0 0 8px; letter-spacing: -0.7px; line-height: 1.15;
}}
#header h1 .accent {{ color: {RED}; }}
#header p {{ color: {MUTED}; font-size: 15px; margin: 0 auto; max-width: 560px; line-height: 1.55; }}

/* ── Thẻ panel chung — phẳng, viền rõ, bóng nhẹ ──────────────────────── */
.panel {{
    background: {PANEL} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 18px !important;
    padding: 22px !important;
    box-shadow: 0 1px 2px rgba(20,20,30,0.04), 0 8px 24px rgba(20,20,30,0.04) !important;
}}
.panel-label {{
    font-weight: 700; font-size: 13.5px; color: {INK};
    margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}}
.panel-label::before {{
    content: ""; width: 7px; height: 7px; border-radius: 50%;
    background: {RED}; flex: none;
}}

/* ── Nút chính — đỏ đặc, không gradient ──────────────────────────────── */
#check-btn {{
    background: {RED} !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 12px !important;
    padding: 12px 18px !important;
    margin-top: 4px !important;
    box-shadow: 0 6px 16px rgba(239,45,60,0.22) !important;
    transition: background .15s ease, transform .05s ease, box-shadow .15s ease;
}}
#check-btn:hover {{ background: {RED_DK} !important; box-shadow: 0 8px 20px rgba(239,45,60,0.30) !important; }}
#check-btn:active {{ transform: translateY(1px); }}

/* ── Ô nhập (font mono cho domain dễ đọc) ────────────────────────────── */
#dom-input textarea {{
    border-radius: 12px !important;
    border: 1px solid {BORDER} !important;
    background: {SUBTLE} !important;
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    color: {INK} !important;
    transition: border-color .15s ease, box-shadow .15s ease, background .15s ease;
}}
#dom-input textarea:focus {{
    border-color: {RED} !important;
    background: #fff !important;
    box-shadow: 0 0 0 4px rgba(239,45,60,0.10) !important;
}}

/* ── Khu kết quả — chiều cao ổn định, cuộn khi nhiều ─────────────────── */
.results {{
    min-height: 360px;
    max-height: 620px;
    overflow-y: auto;
    overflow-x: hidden;
    padding-right: 4px;
}}
.results::-webkit-scrollbar {{ width: 8px; }}
.results::-webkit-scrollbar-thumb {{ background: #DEDEE3; border-radius: 999px; }}
.results::-webkit-scrollbar-track {{ background: transparent; }}

.summary {{ display: flex; gap: 10px; margin-bottom: 18px; }}
.stat {{
    flex: 1; min-width: 0;
    background: {SUBTLE};
    border: 1px solid {BORDER};
    border-radius: 13px;
    padding: 13px 14px;
    display: flex; flex-direction: column; gap: 3px;
}}
.stat-num {{ font-size: 23px; font-weight: 800; color: {INK}; line-height: 1; }}
.stat-lbl {{ font-size: 11px; color: {MUTED}; font-weight: 700;
            text-transform: uppercase; letter-spacing: .045em; }}
.stat.danger {{ background: {DANGER_BG}; border-color: #F6C9CC; }}
.stat.danger .stat-num {{ color: {DANGER}; }}
.stat.safe {{ background: {SAFE_BG}; border-color: #BFEACE; }}
.stat.safe .stat-num {{ color: {SAFE}; }}

.cards {{ display: flex; flex-direction: column; gap: 11px; }}
.card {{
    background: #fff;
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 15px 16px;
    transition: border-color .15s ease, box-shadow .15s ease;
}}
.card:hover {{ box-shadow: 0 4px 14px rgba(20,20,30,0.06); }}
.card.danger {{ border-left: 4px solid {DANGER}; }}
.card.safe   {{ border-left: 4px solid {SAFE}; }}

.card-top {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
.card .domain {{
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
    font-weight: 600; font-size: 14.5px; color: {INK}; word-break: break-all;
}}
.card .badge {{
    font-size: 12px; font-weight: 700; padding: 5px 12px;
    border-radius: 999px; white-space: nowrap; flex: none;
}}
.badge.danger {{ background: {DANGER_BG}; color: {DANGER}; }}
.badge.safe   {{ background: {SAFE_BG}; color: {SAFE}; }}

/* ── Thanh điểm — màu đặc, KHÔNG gradient ────────────────────────────── */
.card .bar {{
    height: 7px; background: #EFEFF2; border-radius: 999px;
    margin: 13px 0 9px; overflow: hidden;
}}
.card .fill {{ height: 100%; border-radius: 999px; transition: width .4s ease; }}
.fill.danger {{ background: {DANGER}; }}
.fill.safe   {{ background: {SAFE}; }}
.card .meta {{ font-size: 12.5px; color: {MUTED}; display: flex; align-items: center; }}
.card .meta b {{ color: {BODY}; font-weight: 700; }}
.card .meta .dot {{ margin: 0 9px; color: #D5D5DA; }}

/* ── Trạng thái rỗng ─────────────────────────────────────────────────── */
.empty-state {{ text-align: center; color: {MUTED}; padding: 96px 20px; }}
.empty-state .empty-icon {{ font-size: 40px; opacity: .4; margin-bottom: 12px; }}
.empty-state p {{ font-size: 14.5px; line-height: 1.65; }}

/* ── Chân trang ──────────────────────────────────────────────────────── */
#appfoot {{
    text-align: center; color: {MUTED}; font-size: 12.5px;
    padding: 26px 0 4px; line-height: 1.6;
}}
#appfoot .dot {{ margin: 0 7px; color: #CFCFD6; }}

/* ── Chỉ giữ MỘT thẻ trắng ──────────────────────────────────────────────
   gr.Group render thành 2 lớp .gr-group.panel lồng nhau + 1 .styler bên
   trong (lớp này mang nền xám mặc định). Làm phẳng tất cả lớp lồng để chỉ
   còn thẻ ngoài cùng, tránh "khung lồng khung" và xóa nền xám. */
.panel .panel, .panel .styler, .panel .gr-group {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    border-radius: 0 !important;
}}
.panel .block, .panel .form,
.panel .html-container, .panel .prose {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

/* ── Slider: nền trong suốt + nhãn gọn (tránh hộp màu lạ) ────────────── */
#thr-slider, #thr-slider .head, #thr-slider .wrap {{ background: transparent !important; }}
#thr-slider {{ margin-top: 2px; }}
#thr-slider label, #thr-slider label > span,
#thr-slider span[data-testid="block-info"] {{
    background: transparent !important; color: {BODY} !important;
    font-weight: 600 !important; font-size: 13px !important;
}}
#thr-slider input[type="number"] {{
    background: {SUBTLE} !important; border: 1px solid {BORDER} !important;
    border-radius: 8px !important; color: {INK} !important;
}}

/* ── Khu ví dụ — thẻ trắng, chữ mono, không tràn ngang ──────────────── */
#ex-wrap {{ margin-top: 18px; }}
#ex-wrap .block, #ex-wrap .form {{
    background: transparent !important; border: none !important; box-shadow: none !important;
}}
#ex-wrap label, #ex-wrap .label, #ex-wrap .label *, #ex-wrap label > span {{
    color: {MUTED} !important; font-weight: 600 !important; font-size: 13px !important;
}}
#ex-wrap table {{
    border: 1px solid {BORDER} !important; border-radius: 12px !important;
    background: #fff !important; overflow: hidden !important;
    width: 100% !important; table-layout: auto !important;
}}
#ex-wrap thead {{ display: none !important; }}
#ex-wrap td {{
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace !important;
    font-size: 12.5px !important; color: {BODY} !important;
    white-space: normal !important; word-break: break-word !important;
    background: #fff !important; border-top: 1px solid {BORDER} !important;
    padding: 10px 13px !important; cursor: pointer;
}}
#ex-wrap tbody tr:first-child td {{ border-top: none !important; }}
#ex-wrap tbody tr:hover td {{ background: {SUBTLE} !important; }}
/* dự phòng nếu Gradio render ví dụ dạng nút thay vì bảng */
#ex-wrap [class*="sample"] {{
    background: #fff !important; border: 1px solid {BORDER} !important;
    border-radius: 10px !important; color: {BODY} !important;
    font-family: ui-monospace, Menlo, Consolas, monospace !important; font-size: 12.5px !important;
}}

/* ── Responsive ──────────────────────────────────────────────────────── */
@media (max-width: 640px) {{
    #header {{ padding-top: 28px; }}
    #header h1 {{ font-size: 28px; }}
    .panel {{ padding: 18px !important; }}
    .summary {{ flex-wrap: wrap; }}
    .stat {{ flex: 1 1 30%; }}
}}
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill=PAGE,
    block_background_fill=PANEL,
    block_border_color=BORDER,
    block_radius="18px",
    block_shadow="0 1px 2px rgba(20,20,30,0.04)",
    input_background_fill=SUBTLE,
    input_border_color=BORDER,
    input_radius="12px",
    button_large_radius="12px",
    button_primary_background_fill=RED,
    button_primary_background_fill_hover=RED_DK,
    button_primary_text_color="#FFFFFF",
    slider_color=RED,
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
                    elem_id="thr-slider",
                )
                btn = gr.Button("🔎  Kiểm tra", variant="primary", elem_id="check-btn")

            with gr.Column(elem_id="ex-wrap"):
                gr.Examples(
                    examples=[
                        ["google.com\npaypal.com\ngithub.com"],
                        ["mortiscontrastatim.com\njeannetteabrahamson.net\nthinkhomereach.art"],
                        ["txumyqrubwutbb.cc\nbelievekeylab.in"],
                    ],
                    inputs=[inp],
                    label="Ví dụ — bấm để điền nhanh",
                )

        with gr.Column(scale=6):
            with gr.Group(elem_classes="panel"):
                gr.HTML('<span class="panel-label">Kết quả phân tích</span>')
                out = gr.HTML(EMPTY_HTML, elem_classes="results")

    gr.HTML(
        '<div id="appfoot">Bilbo DGA Detector'
        '<span class="dot">·</span>CNN + LSTM ensemble'
        '<span class="dot">·</span>chỉ dùng cho mục đích nghiên cứu &amp; học tập</div>'
    )

    btn.click(check_domains, inputs=[inp, threshold], outputs=out)
    inp.submit(check_domains, inputs=[inp, threshold], outputs=out)


if __name__ == '__main__':
    demo.launch(theme=theme, css=CSS, head=FORCE_LIGHT_HEAD)
