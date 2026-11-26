"""
Ứng dụng Demo Nhận diện Cảm xúc Khuôn mặt (Facial Emotion Recognition - FER2013)
Bài tập lớn 02 - Môn Học sâu (Deep Learning)
Nhóm thực hiện: Tuyết Nhi, Phú Dương, Xuân Anh, Quốc Việt
"""

import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path để import từ src
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import io
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F

from src.models import BaselineCNN, SmallCNN, Medium01CNN, VGG16_FailCase

# -----------------------------------------------------------------------------
# Cấu hình giao diện Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nhận diện Cảm xúc Khuôn mặt - FER2013",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Hệ thống giao diện: bảng điều khiển tối, nhấn sắc tím/xanh ngọc.
# Phần xử lý ảnh vẫn dùng các component Streamlit chuẩn để giữ độ ổn định.
# -----------------------------------------------------------------------------
st.markdown("""<style>
:root{--ink:#f8fbff;--muted:#9caec8;--panel:rgba(16,29,52,.72);--line:rgba(154,181,225,.16);--aqua:#58e0d1;--violet:#9e7bff}
html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 12% -10%,rgba(88,224,209,.16),transparent 30%),radial-gradient(circle at 88% 0%,rgba(158,123,255,.20),transparent 34%),#070d1a;color:var(--ink)}
[data-testid="stHeader"]{background:rgba(7,13,26,.75)}
.block-container{box-sizing:border-box!important;width:100%!important;max-width:1380px!important;margin-left:auto!important;margin-right:auto!important;padding:3.5rem 2.2rem 3.5rem!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1930 0%,#08101f 100%);border-right:1px solid var(--line)}
[data-testid="stSidebar"]>div:first-child{padding-top:1.3rem}
[data-testid="stSidebar"] *{color:#e8effd}
[data-testid="stSidebar"] [data-testid="stSelectbox"]>div>div,[data-testid="stSidebar"] [data-testid="stSlider"]{background:rgba(255,255,255,.035)}
h1,h2,h3,p{color:var(--ink)}
.brand-shell{position:relative;overflow:hidden;margin:0 0 1.4rem;padding:1.7rem 1.9rem;border:1px solid rgba(178,203,244,.20);border-radius:24px;background:linear-gradient(115deg,rgba(18,39,68,.92),rgba(17,26,55,.84));box-shadow:0 26px 70px rgba(0,0,0,.25)}
.brand-shell::after{content:"";position:absolute;width:260px;height:260px;right:-70px;top:-160px;border-radius:50%;background:radial-gradient(circle,rgba(88,224,209,.34),transparent 68%);pointer-events:none}
.eyebrow{color:#7eeade;font-size:.72rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;margin-bottom:.45rem}
.brand-shell h1.hero-title{display:block;max-width:100%;margin:0!important;padding:0!important;color:#fff!important;font-size:clamp(1.5rem,3.5vw,2.4rem);line-height:1.15!important;letter-spacing:-.035em;font-weight:800}
.hero-subtitle{max-width:680px;margin:.6rem 0 1rem;color:#b7c8e4;line-height:1.6;font-size:.93rem}
.hero-stats{display:flex;flex-wrap:wrap;gap:.5rem;position:relative;z-index:1;margin-top:.5rem}
.hero-stat{display:inline-flex;align-items:center;gap:.4rem;border-radius:999px;font-size:.75rem;font-weight:700;padding:.28rem .7rem;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);color:#ccdaee}
.workspace-title{margin:.15rem 0 .28rem;color:#fff;font-size:1.08rem;font-weight:700;line-height:1.35}
.workspace-copy{margin:0 0 1rem;color:var(--muted);font-size:.9rem;line-height:1.55}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:.35rem;border-bottom:1px solid var(--line);margin-bottom:1.1rem}
[data-testid="stTabs"] button[role="tab"]{height:42px;border-radius:10px 10px 0 0;padding:0 .8rem;color:#9aacc7;font-weight:700}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:#8df0e3;background:rgba(88,224,209,.08)}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:#58e0d1;height:2px}
[data-testid="stCameraInput"]{border:1px dashed rgba(130,220,220,.4)!important;border-radius:18px!important;background:linear-gradient(140deg,rgba(88,224,209,.07),rgba(158,123,255,.055))!important;padding:.7rem!important;max-width:820px;margin:.45rem auto 0;overflow:hidden}
[data-testid="stCameraInput"] video{width:100%!important;height:min(56.25vw,460px)!important;max-height:460px!important;object-fit:cover!important;border-radius:12px!important}
[data-testid="stCameraInput"] img{border-radius:12px!important}
[data-testid="stFileUploader"]{max-width:640px}
[data-testid="stFileUploaderDropzone"]{border:1px dashed rgba(130,220,220,.4)!important;border-radius:18px!important;background:linear-gradient(140deg,rgba(88,224,209,.07),rgba(158,123,255,.055))!important;padding:.8rem 1rem!important;min-height:80px!important;display:flex!important;align-items:center;justify-content:flex-start}
[data-testid="stFileUploaderDropzone"]>div{width:100%;background:transparent!important}
[data-testid="stFileUploaderDropzoneInstructions"]{background:transparent!important}
[data-testid="stFileUploaderDropzoneInstructions"]>div{background:transparent!important;border:none!important;padding:0!important}
[data-testid="stFileUploaderDropzone"] small,[data-testid="stFileUploaderDropzone"] span,[data-testid="stFileUploaderDropzone"] p{color:#aebfdb!important;background:transparent!important}
[data-testid="stFileUploaderDropzone"] svg{fill:#58e0d1!important;opacity:.7}
[data-testid="stFileUploaderDropzone"] button[title="Add files"],[data-testid="stFileUploader"] button[title="Add files"]{display:none!important}
[data-testid="stFileUploaderFile"]{background:rgba(255,255,255,.05)!important;border:1px solid rgba(154,181,225,.2)!important;border-radius:10px!important}
[data-testid="stFileUploaderFile"] span,[data-testid="stFileUploaderFile"] small{color:#c8d8f0!important}
[data-testid="stCameraInput"] button,.stButton>button{border:0!important;border-radius:11px!important;background:linear-gradient(135deg,#50d5c6,#7185fa)!important;color:#06101d!important;font-weight:800!important;box-shadow:0 9px 22px rgba(88,224,209,.15);transition:transform .18s ease,box-shadow .18s ease}
[data-testid="stCameraInput"] button:hover,.stButton>button:hover{transform:translateY(-1px);box-shadow:0 13px 28px rgba(88,224,209,.23)}
[data-testid="stCameraInput"] button *,[data-testid="stCameraInput"] button p{color:#06101d!important}
[data-testid="stCameraInput"] button p{font-size:0!important}
[data-testid="stCameraInput"] button p::after{content:"📸 Chụp ảnh";display:block;color:#06101d!important;font-size:.9rem!important;font-weight:800}
[data-testid="stCameraInput"] button{display:flex!important;align-items:center!important;justify-content:center!important;font-size:0!important}
[data-testid="stCameraInput"] button>*{font-size:0!important}
[data-testid="stCameraInput"] button::after{content:"📸 Chụp ảnh";color:#06101d!important;font-size:.9rem!important;font-weight:800}
[data-testid="stFileUploaderDropzone"] button{border:0!important;border-radius:11px!important;background:linear-gradient(135deg,#50d5c6,#7185fa)!important;color:#06101d!important;font-weight:800!important;box-shadow:0 9px 22px rgba(88,224,209,.15);transition:transform .18s ease,box-shadow .18s ease}
[data-testid="stFileUploaderDropzone"] button:hover{transform:translateY(-1px);box-shadow:0 13px 28px rgba(88,224,209,.23)}
[data-testid="stImage"] img{border:1px solid rgba(159,188,234,.18);border-radius:17px;box-shadow:0 18px 45px rgba(0,0,0,.20)}
.result-panel{height:100%;padding:1.05rem;border:1px solid var(--line);border-radius:18px;background:var(--panel)}
.prediction-card{position:relative;overflow:hidden;padding:1.1rem;border:1px solid rgba(96,220,211,.26);border-radius:15px;background:linear-gradient(135deg,rgba(58,181,173,.17),rgba(111,104,229,.15))}
.prediction-card::after{content:"";position:absolute;width:150px;height:150px;right:-58px;bottom:-95px;border-radius:50%;background:rgba(136,119,255,.18)}
.prediction-header{display:flex;align-items:center;justify-content:space-between;gap:.5rem;margin-bottom:.5rem}
.prediction-kicker{color:#b9d4f7;font-size:.7rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase}
.prediction-label{display:flex;align-items:center;gap:.62rem;font-size:1.5rem;font-weight:800}
.confidence-badge{position:relative;z-index:1;padding:.3rem .55rem;border-radius:999px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.17);color:#fff;font-size:.75rem;font-weight:800;white-space:nowrap}
.emotion-heading{margin:1rem 0 .55rem;color:#dce8fc;font-size:.82rem;font-weight:800;letter-spacing:.05em;text-transform:uppercase}
.emotion-row{display:flex;align-items:center;gap:.45rem;padding:.27rem .46rem;margin-bottom:.15rem;border:1px solid transparent;border-radius:10px;line-height:1.2;transition:background .18s ease}
.emotion-row:hover{background:rgba(255,255,255,.04)}
.emotion-row.top-prediction{background:rgba(88,224,209,.085);border-color:rgba(88,224,209,.18)}
.emotion-name{width:140px;color:#e5eefc;font-size:.8rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.progress-track{flex:1;height:7px;overflow:hidden;border-radius:99px;background:rgba(198,216,246,.12)}
.progress-fill{height:100%;border-radius:inherit}
.emotion-value{width:44px;color:#d9e5f8;text-align:right;font-size:.75rem;font-weight:750;font-variant-numeric:tabular-nums}
.guide-card{height:calc(100% - .75rem);padding:1.1rem 1.15rem;border:1px solid var(--line);border-radius:17px;background:var(--panel);color:#c2d0e6}
.guide-card b{color:#fff}.guide-card li{margin:.4rem 0}
.status-badge{display:inline-flex;align-items:center;gap:.42rem;margin-bottom:.85rem;padding:.4rem .65rem;border-radius:999px;background:rgba(88,224,209,.12);border:1px solid rgba(88,224,209,.22);color:#9df2e7;font-size:.77rem;font-weight:700}
.status-dot{width:7px;height:7px;border-radius:50%;background:#58e0d1;box-shadow:0 0 0 4px rgba(88,224,209,.1)}
.sample-label{color:#c6d5ec;font-size:.82rem;font-weight:750;text-align:center;margin:.45rem 0 .35rem}
div[data-testid="stAlert"],div[data-testid="stAlert"]>div,div[data-testid="stAlert"] [role="alert"]{border-radius:13px!important;border:1px solid rgba(245,158,11,.45)!important;background:rgba(97,57,4,.72)!important;color:#fff3c4!important;box-shadow:none!important}
div[data-testid="stAlert"] *{color:#fff3c4!important}
iframe[height="0"]{display:none!important}
@media(max-width:640px){.block-container{padding:1rem .9rem 2rem!important}.brand-shell{padding:1.35rem;border-radius:18px}.emotion-name{width:118px}.hero-title{font-size:1.5rem!important}[data-testid="stCameraInput"] video{height:min(56.25vw,300px)!important}}
</style>""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Danh mục nhãn và ánh xạ cảm xúc FER2013
# -----------------------------------------------------------------------------
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

EMOTION_INFO = {
    'happy': {'vi': 'Hạnh phúc', 'en': 'Happy', 'emoji': '😊', 'color': '#F59E0B', 'gradient': 'linear-gradient(90deg, #FBBF24, #F59E0B)'},
    'surprise': {'vi': 'Ngạc nhiên', 'en': 'Surprise', 'emoji': '😲', 'color': '#06B6D4', 'gradient': 'linear-gradient(90deg, #22D3EE, #06B6D4)'},
    'neutral': {'vi': 'Bình thường', 'en': 'Neutral', 'emoji': '😐', 'color': '#D5E1F2', 'gradient': 'linear-gradient(90deg, #B9C8DA, #8095AD)'},
    'sad': {'vi': 'Buồn bã', 'en': 'Sad', 'emoji': '😢', 'color': '#3B82F6', 'gradient': 'linear-gradient(90deg, #60A5FA, #3B82F6)'},
    'angry': {'vi': 'Tức giận', 'en': 'Angry', 'emoji': '😡', 'color': '#EF4444', 'gradient': 'linear-gradient(90deg, #F87171, #EF4444)'},
    'fear': {'vi': 'Sợ hãi', 'en': 'Fear', 'emoji': '😨', 'color': '#8B5CF6', 'gradient': 'linear-gradient(90deg, #A78BFA, #8B5CF6)'},
    'disgust': {'vi': 'Ghê tởm', 'en': 'Disgust', 'emoji': '🤢', 'color': '#10B981', 'gradient': 'linear-gradient(90deg, #34D399, #10B981)'}
}

YUNET_MODEL_PATH = ROOT_DIR / "demo/face_detection_yunet.onnx"

# -----------------------------------------------------------------------------
# Quản lý tải mô hình & Checkpoint (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_face_detector():
    """Tải bộ phát hiện khuôn mặt YuNet (Deep Learning ONNX) của OpenCV."""
    if not YUNET_MODEL_PATH.is_file():
        st.error(f"Không tìm thấy file mô hình YuNet tại: {YUNET_MODEL_PATH}")
        return None
    detector = cv2.FaceDetectorYN.create(
        model=str(YUNET_MODEL_PATH),
        config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )
    return detector


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, device: str = "cpu"):
    """
    Tải mô hình BaselineCNN và nạp trọng số đã huấn luyện.
    Hỗ trợ cả state_dict trực tiếp lẫn dictionary có bọc key.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        return None, f"Không tìm thấy file trọng số tại: {path}"

    try:
        path_str = str(path).lower()
        if "vgg16" in path_str:
            model = VGG16_FailCase(num_classes=7)
        elif "small" in path_str:
            model = SmallCNN(num_classes=7)
        elif "medium" in path_str:
            model = Medium01CNN(num_classes=7)
        else:
            model = BaselineCNN(num_classes=7)

        payload = torch.load(path, map_location=device, weights_only=True)
        if isinstance(payload, dict):
            state = payload.get("model_state_dict", payload.get("state_dict", payload))
        else:
            state = payload

        if isinstance(state, dict):
            state = {k.removeprefix("module."): v for k, v in state.items()}

        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Lỗi khi nạp mô hình: {str(e)}"


def detect_faces_yunet(detector, img_bgr: np.ndarray, score_threshold: float = 0.6):
    """Phát hiện khuôn mặt sử dụng YuNet detector."""
    h, w = img_bgr.shape[:2]
    detector.setInputSize((w, h))
    detector.setScoreThreshold(score_threshold)
    _, faces = detector.detect(img_bgr)
    results = []
    if faces is not None:
        for face in faces:
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            conf = float(face[14])
            x = max(0, x)
            y = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)
            if fw > 10 and fh > 10:
                results.append((x, y, fw, fh, conf))
    return results


def preprocess_face(face_gray: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Tiền xử lý ảnh khuôn mặt theo chuẩn FER2013: 48x48, normalize [-1, 1]."""
    resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 48, 48]
    return tensor.to(device)


def predict_emotion(model, face_tensor: torch.Tensor):
    """Thực hiện suy luận và tính xác suất cho 7 biểu cảm."""
    with torch.inference_mode():
        logits = model(face_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probabilities))
    return pred_idx, probabilities


# -----------------------------------------------------------------------------
# Vẽ khung nhận diện bằng PIL (Hỗ trợ 100% tiếng Việt)
# -----------------------------------------------------------------------------
def draw_face_annotations(img_rgb: np.ndarray, face_boxes_with_labels: list) -> np.ndarray:
    """Vẽ bounding box và nhãn tiếng Việt sắc nét, không bị lỗi font."""
    annotated = img_rgb.copy()
    
    for (x, y, w, h, label_text) in face_boxes_with_labels:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (16, 185, 129), 2)
        corner_len = max(8, min(w, h) // 6)
        cv2.line(annotated, (x, y), (x + corner_len, y), (5, 150, 105), 3)
        cv2.line(annotated, (x, y), (x, y + corner_len), (5, 150, 105), 3)
        cv2.line(annotated, (x + w, y), (x + w - corner_len, y), (5, 150, 105), 3)
        cv2.line(annotated, (x + w, y), (x + w, y + corner_len), (5, 150, 105), 3)
        cv2.line(annotated, (x, y + h), (x + corner_len, y + h), (5, 150, 105), 3)
        cv2.line(annotated, (x, y + h), (x, y + h - corner_len), (5, 150, 105), 3)
        cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), (5, 150, 105), 3)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), (5, 150, 105), 3)

    pil_img = Image.fromarray(annotated)
    draw = ImageDraw.Draw(pil_img)

    font = None
    for font_path in ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/tahoma.ttf"]:
        if Path(font_path).is_file():
            try:
                font = ImageFont.truetype(font_path, size=14)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    for (x, y, w, h, label_text) in face_boxes_with_labels:
        bbox = draw.textbbox((x, y), label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        tag_y0 = y - text_h - 8 if y - text_h - 8 >= 0 else y + 3
        tag_y1 = tag_y0 + text_h + 5
        tag_x1 = min(pil_img.width, x + text_w + 12)

        draw.rectangle([x, tag_y0, tag_x1, tag_y1], fill=(16, 185, 129))
        draw.text((x + 6, tag_y0 + 2), label_text, font=font, fill=(255, 255, 255))

    return np.array(pil_img)


# -----------------------------------------------------------------------------
# Thanh công cụ bên cạnh (Sidebar)
# -----------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.markdown("""
    <div class="status-badge">
        <span class="status-dot"></span> LIVE · Hệ thống sẵn sàng
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("### Bảng điều khiển")
    st.sidebar.caption("Tùy chỉnh mô hình và độ nhạy phát hiện.")

    st.sidebar.markdown("#### 🧠 Mô hình suy luận")
    available_checkpoints = {
        "Baseline CNN (Đề xuất)": ROOT_DIR / "checkpoints/baseline_best.pth",
    }

    extra_ckpt_map = {
        "Small CNN (Thử nghiệm)": ROOT_DIR / "checkpoints/small_best.pth",
        "Medium-01 CNN (Thử nghiệm)": ROOT_DIR / "checkpoints/medium_best.pth",
        "VGG16 Custom (Thử nghiệm)": ROOT_DIR / "checkpoints/vgg16_best.pth",
    }
    for label, p in extra_ckpt_map.items():
        if p.exists():
            available_checkpoints[label] = p

    selected_model_name = st.sidebar.selectbox(
        "Chọn checkpoint",
        options=list(available_checkpoints.keys()),
        index=0
    )
    checkpoint_path = str(available_checkpoints[selected_model_name])

    # Thiết bị phần cứng
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device_label = f"GPU ({torch.cuda.get_device_name(0)})" if device_name == "cuda" else "CPU thông thường"
    st.sidebar.markdown(f"<div class='model-pill'>⚡ {device_label}</div>", unsafe_allow_html=True)

    with st.sidebar.expander("📊 Thông số kỹ thuật", expanded=False):
        if "vgg16" in checkpoint_path.lower():
            st.markdown("""
            - **Kiến trúc:** VGG16 Custom (Fail Case)
            - **Kích thước ảnh:** 48 × 48 px (Grayscale)
            - **Feature Map cuối:** 1 × 1 × 512 (bị teo kích thước)
            - **Tổng tham số:** ~134.3M (Quá khớp nặng)
            - **Độ chính xác Test:** **~44.30%** *(thấp hơn Baseline)*
            """)
        elif "small" in checkpoint_path.lower():
            st.markdown("""
            - **Kiến trúc:** Small CNN (2 Conv block)
            - **Kích thước ảnh:** 48 × 48 px (Grayscale)
            - **Feature Map cuối:** 12 × 12 × 256
            - **Tổng tham số:** ~2.5M
            """)
        elif "medium" in checkpoint_path.lower():
            st.markdown("""
            - **Kiến trúc:** Medium-01 CNN (3 Conv block sâu)
            - **Kích thước ảnh:** 48 × 48 px (Grayscale)
            - **Feature Map cuối:** 6 × 6 × 256
            - **Tổng tham số:** ~5.8M
            """)
        else:
            st.markdown("""
            - **Kiến trúc:** 3 Block Conv2D-BN-ReLU-MaxPool
            - **Kích thước ảnh:** 48 × 48 px (Grayscale)
            - **Feature Map cuối:** 6 × 6 × 128
            - **Tổng tham số:** 1.274.823 (~1.27M)
            - **Độ chính xác Test:** **48.41%**
            """)

    st.sidebar.markdown("#### 🎯 Phát hiện khuôn mặt")
    score_threshold = st.sidebar.slider(
        "Độ nhạy YuNet",
        min_value=0.3,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="Giảm ngưỡng khi phòng thiếu sáng hoặc mặt ở góc nghiêng"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("FER2013 · 7 cảm xúc · Deep Learning")
    st.sidebar.caption("Bài tập lớn 02 · Học sâu")

    return checkpoint_path, selected_model_name, device_name, score_threshold


# -----------------------------------------------------------------------------
# Render Thanh Tiến trình Cảm xúc Tinh gọn (Chuẩn hóa tránh lỗi Markdown Code Block)
# -----------------------------------------------------------------------------
def render_custom_emotion_bars(probs: np.ndarray, pred_idx: int):
    """Render 7 thanh cảm xúc nhỏ gọn, nối chuỗi trực tiếp để không bị lỗi thụt lề Markdown."""
    sorted_indices = np.argsort(probs)[::-1]

    rows = []
    for idx in sorted_indices:
        emotion_key = CLASSES[idx]
        info = EMOTION_INFO[emotion_key]
        percentage = probs[idx] * 100
        is_top = (idx == pred_idx)

        row_class = "emotion-row top-prediction" if is_top else "emotion-row"
        badge_star = "★ " if is_top else ""

        row_html = (
            f'<div class="{row_class}">'
            f'<div class="emotion-name" title="{info["vi"]} ({info["en"]})">'
            f'<span style="font-size: 1.05rem;">{info["emoji"]}</span>'
            f'<span>{badge_star}{info["vi"]} <small style="opacity: 0.65;">({info["en"]})</small></span>'
            f'</div>'
            f'<div class="progress-track">'
            f'<div class="progress-fill" style="width: {percentage:.1f}%; background: {info["gradient"]};"></div>'
            f'</div>'
            f'<div class="emotion-value" style="color: {info["color"]};">{percentage:.1f}%</div>'
            f'</div>'
        )
        rows.append(row_html)

    full_html = '<div style="margin-top: 0.5rem;">' + "".join(rows) + '</div>'
    st.markdown(full_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Giao diện chính và xử lý ảnh
# -----------------------------------------------------------------------------
def process_and_display(image: Image.Image, model, detector, score_threshold: float, device: str):
    """Phát hiện khuôn mặt, suy luận cảm xúc và hiển thị kết quả cạnh ảnh nguồn."""
    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = detect_faces_yunet(detector, img_bgr, score_threshold=score_threshold)

    col_img, col_result = st.columns([1.05, 1], gap="large")

    results = []
    face_boxes_with_labels = []

    if len(faces) == 0:
        with col_img:
            st.warning("Chưa phát hiện thấy khuôn mặt rõ ràng trong ảnh.")
            analyze_full = st.checkbox("Vẫn phân tích toàn bộ ảnh", value=True, help="Hữu ích với ảnh FER2013 có khuôn mặt cận cảnh.")
            if analyze_full:
                face_tensor = preprocess_face(img_gray, device=device)
                pred_idx, probs = predict_emotion(model, face_tensor)
                results.append((0, 0, img_gray.shape[1], img_gray.shape[0], pred_idx, probs))
            st.image(image, caption="Ảnh nguồn", width="stretch")
    else:
        for (x, y, w, h, face_conf) in faces:
            face_roi = img_gray[y:y+h, x:x+w]
            face_tensor = preprocess_face(face_roi, device=device)
            pred_idx, probs = predict_emotion(model, face_tensor)
            results.append((x, y, w, h, pred_idx, probs))

            emotion_key = CLASSES[pred_idx]
            info = EMOTION_INFO[emotion_key]
            label_text = f"{info['vi']} ({probs[pred_idx]:.1%})"
            face_boxes_with_labels.append((x, y, w, h, label_text))

        annotated_rgb = draw_face_annotations(img_rgb, face_boxes_with_labels)

        with col_img:
            st.image(annotated_rgb,
                     caption=f"YuNet đã phát hiện {len(faces)} khuôn mặt",
                     width="stretch")

    with col_result:
        if not results:
            st.info("Hãy dùng ảnh rõ mặt hoặc giảm độ nhạy YuNet trong bảng điều khiển.")
            return

        st.markdown("<p class='workspace-title'>Kết quả phân tích</p><p class='workspace-copy'>Xác suất được sắp xếp từ cao đến thấp.</p>", unsafe_allow_html=True)
        for idx, (x, y, w, h, pred_idx, probs) in enumerate(results):
            pred_emotion = CLASSES[pred_idx]
            info = EMOTION_INFO[pred_emotion]

            if len(results) > 1:
                st.markdown(f"**👤 Khuôn mặt #{idx + 1}**")

            # Thẻ kết quả nổi bật tinh gọn (viết thẳng không thụt lề tránh lỗi Markdown)
            card_html = (
                f'<div class="prediction-card">'
                f'<div class="prediction-header">'
                f'<span class="prediction-kicker">Dự đoán nổi bật</span>'
                f'<span class="confidence-badge">Tin cậy: {probs[pred_idx] * 100:.1f}%</span>'
                f'</div>'
                f'<div class="prediction-label" style="color: {info["color"]};">'
                f'<span style="font-size: 1.75rem;">{info["emoji"]}</span>'
                f'<span>{info["vi"]} <span style="font-size: 0.92rem; font-weight: 600; opacity: 0.75;">({info["en"]})</span></span>'
                f'</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

            st.markdown("<p class='emotion-heading'>Phân bố 7 cảm xúc</p>", unsafe_allow_html=True)
            render_custom_emotion_bars(probs, pred_idx)


# -----------------------------------------------------------------------------
# Điểm vào chính của ứng dụng
# -----------------------------------------------------------------------------
def main():
    checkpoint_path, model_name, device, score_threshold = render_sidebar()

    st.markdown(
        f'<section class="brand-shell">'
        f'<div class="eyebrow">🎭 Facial Emotion Recognition · FER-2013</div>'
        f'<h1 class="hero-title">Demo Nhận Diện Cảm Xúc Khuôn Mặt</h1>'
        f'<p class="hero-subtitle">Nhận diện <strong>7 biểu cảm</strong> thời gian thực bằng mô hình Deep Learning CNN trên tập dữ liệu FER-2013.</p>'
        f'<div class="hero-stats">'
        f'<span class="hero-stat">🎭 7 cảm xúc</span>'
        f'<span class="hero-stat">◉ YuNet Face Detector</span>'
        f'<span class="hero-stat">🧠 {model_name}</span>'
        f'</div>'
        f'</section>',
        unsafe_allow_html=True,
    )

    detector = load_face_detector()
    if detector is None:
        st.error("Không thể khởi tạo bộ phát hiện khuôn mặt.")
        st.stop()

    model, err = load_model(checkpoint_path, device=device)
    if err:
        st.error(f"❌ {err}")
        st.stop()

    # Các Tabs điều hướng
    tab_webcam, tab_upload, tab_samples, tab_info = st.tabs([
        "Webcam",
        "Tải ảnh",
        "Ảnh mẫu",
        "Cách dùng"
    ])

    with tab_webcam:
        st.markdown(
            "<p class='workspace-title'>Quét trực tiếp từ webcam</p>"
            "<p class='workspace-copy'>Giữ khuôn mặt ở trung tâm khung hình, đủ sáng và nhìn rõ camera.</p>",
            unsafe_allow_html=True,
        )

        # Lắng nghe phím Space để kích hoạt chụp ảnh hoặc chụp lại
        components.html(
            """
            <script>
            (function() {
                const parentDoc = window.parent.document;
                if (!window.parent.__fer_space_listener_added) {
                    window.parent.__fer_space_listener_added = true;
                    parentDoc.addEventListener('keydown', function(e) {
                        if (e.code === 'Space' || e.key === ' ') {
                            const activeEl = parentDoc.activeElement;
                            const tag = activeEl ? activeEl.tagName : '';
                            if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag) || (activeEl && activeEl.isContentEditable)) {
                                return;
                            }
                            // 1. Nếu camera đang mở: bấm chụp ảnh
                            const camBtn = parentDoc.querySelector('[data-testid="stCameraInput"] button');
                            if (camBtn) {
                                e.preventDefault();
                                camBtn.click();
                                return;
                            }
                            // 2. Nếu đã chụp xong: bấm chụp lại
                            const buttons = Array.from(parentDoc.querySelectorAll('button'));
                            const retakeBtn = buttons.find(b => b.innerText && b.innerText.includes('Chụp ảnh khác'));
                            if (retakeBtn) {
                                e.preventDefault();
                                retakeBtn.click();
                                return;
                            }
                        }
                    });
                }
            })();
            </script>
            """,
            height=0,
            width=0,
        )

        # Kiểm tra trạng thái đã chụp hay chưa
        has_photo = ("webcam_photo" in st.session_state and st.session_state["webcam_photo"] is not None)

        if not has_photo:
            # TRẠNG THÁI 1: HIỂN THỊ CAMERA LIVE ĐỂ CHỤP
            st.caption('Bấm nút **"Chụp ảnh"** hoặc nhấn phím **Space** để chụp.')
            camera_photo = st.camera_input(
                "Chụp ảnh khuôn mặt",
                key="webcam_input_fixed",
                label_visibility="collapsed"
            )
            if camera_photo is not None:
                st.session_state["webcam_photo"] = camera_photo.getvalue()
                st.rerun()
        else:
            # TRẠNG THÁI 2: ĐÃ CHỤP XONG -> PHẦN PHÂN TÍCH ẢNH THAY THẾ HOÀN TOÀN KHUNG CAMERA
            col_retake, _ = st.columns([.82, 1.18], gap="medium")
            with col_retake:
                if st.button("↻ Chụp ảnh khác", type="primary", width="stretch", key="btn_retake_cam"):
                    st.session_state["webcam_photo"] = None
                    st.session_state.pop("webcam_input_fixed", None)
                    st.rerun()

            # Hiển thị ảnh ĐÃ VẼ KHUNG NHẬN DIỆN và bảng phân tích ngay tại vị trí này
            img = Image.open(io.BytesIO(st.session_state["webcam_photo"]))
            process_and_display(img, model, detector, score_threshold, device)

    with tab_upload:
        st.markdown("<p class='workspace-title'>Tải ảnh để phân tích</p><p class='workspace-copy'>Ảnh chân dung chính diện sẽ cho kết quả ổn định nhất. Hỗ trợ JPG, PNG, WEBP và BMP.</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Kéo thả ảnh vào đây hoặc duyệt tệp",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            st.markdown("---")
            img = Image.open(uploaded_file)
            process_and_display(img, model, detector, score_threshold, device)

    with tab_samples:
        st.markdown("<p class='workspace-title'>Khám phá bằng ảnh mẫu</p><p class='workspace-copy'>Thử nhanh các ảnh được chọn từ tập kiểm thử FER2013 để xem luồng nhận diện.</p>", unsafe_allow_html=True)
        sample_candidates = {
            "😊 Hạnh phúc": ROOT_DIR / "data/raw/test/happy/PrivateTest_10077120.jpg",
            "😢 Buồn bã": ROOT_DIR / "data/raw/test/sad/PrivateTest_10247676.jpg",
            "😲 Ngạc nhiên": ROOT_DIR / "data/raw/test/surprise/PrivateTest_10072988.jpg",
            "😡 Tức giận": ROOT_DIR / "data/raw/test/angry/PrivateTest_10131363.jpg"
        }
        
        cols = st.columns(4)
        selected_sample_path = None
        for i, (label, p) in enumerate(sample_candidates.items()):
            with cols[i]:
                if p.is_file():
                    thumb = Image.open(p)
                    st.image(thumb, width="stretch")
                    st.markdown(f"<div class='sample-label'>{label}</div>", unsafe_allow_html=True)
                    if st.button("Phân tích", key=f"btn_sample_{i}", width="stretch"):
                        selected_sample_path = p
                else:
                    st.info(label)

        if selected_sample_path and selected_sample_path.is_file():
            st.markdown("---")
            sample_img = Image.open(selected_sample_path)
            process_and_display(sample_img, model, detector, score_threshold, device)

    with tab_info:
        st.markdown("<p class='workspace-title'>Mẹo để có kết quả tốt</p><p class='workspace-copy'>Một vài lưu ý ngắn trước khi bắt đầu phiên phân tích.</p>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2, gap="medium")
        with col_g1:
            st.markdown("""
            <div class="guide-card">
                <b>📷 Chuẩn bị ảnh đầu vào</b>
                <ul style="margin-top: 0.35rem; padding-left: 1.2rem; font-size: 0.85rem; line-height: 1.5;">
                    <li>Cho phép trình duyệt truy cập camera máy tính.</li>
                    <li>Căn chỉnh khuôn mặt vào giữa khung, đủ sáng.</li>
                    <li>Nhấn nút chụp ảnh hoặc nhấn phím Space để bắt đầu nhận diện.</li>
                    <li>Ảnh tải lên nên có kích cỡ <b>400×400 đến 1000×1000 px</b>.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col_g2:
            st.markdown("""
            <div class="guide-card">
                <b>📊 Đọc kết quả & xử lý sự cố</b>
                <ul style="margin-top: 0.35rem; padding-left: 1.2rem; font-size: 0.85rem; line-height: 1.5;">
                    <li><b>Khung nhận diện:</b> vị trí mặt kèm nhãn và độ tin cậy.</li>
                    <li><b>Thanh xác suất:</b> cho biết mức độ tin cậy tương đối giữa 7 cảm xúc.</li>
                    <li>Nếu chưa thấy mặt: giảm <b>độ nhạy YuNet</b> xuống khoảng <b>0.40 – 0.50</b>.</li>
                    <li>Tránh ánh sáng ngược từ sau lưng để AI bắt nét tốt nhất.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
