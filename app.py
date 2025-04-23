import os, gc, io, base64, pathlib
import cv2
import numpy as np
from PIL import Image

import streamlit as st

st.set_page_config(
    page_title="Document Scanner",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed",
)


from streamlit_drawable_canvas import st_canvas
import torch
import torchvision.transforms as torchvision_T
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large,
)


@st.cache_resource(show_spinner="🔄 Đang tải model …")
def load_model(num_classes: int = 2, model_name: str = "mbv3", device: torch.device | None = None):
    """Tải và cache model segmentation.
    Args:
        num_classes: số lớp output.
        model_name: "mbv3" hoặc "r50".
        device: thiết bị PyTorch. Nếu None sẽ tự động chọn CUDA nếu khả dụng.
    Returns:
        Model đã đặt ở eval mode trên đúng device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "mbv3":
        model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True)
        ckpt = "model_mbv3_iou_mix_2C049.pth"
    else:
        model = deeplabv3_resnet50(num_classes=num_classes, aux_loss=True)
        ckpt = "model_r50_iou_mix_2C020.pth"

    checkpoint_path = os.path.join(os.getcwd(), ckpt)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

    # warm‑up để tối ưu JIT/CUDA graph lần đầu
    _ = model(torch.randn(1, 3, 384, 384, device=device))
    return model



def image_preprocess_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    return torchvision_T.Compose(
        [torchvision_T.ToTensor(), torchvision_T.Normalize(mean, std)]
    )


def order_points(pts):
    """Sắp xếp lại 4 điểm theo thứ tự TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype(int).tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(np.array(br) - np.array(bl))
    widthB = np.linalg.norm(np.array(tr) - np.array(tl))
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(np.array(tr) - np.array(br))
    heightB = np.linalg.norm(np.array(tl) - np.array(bl))
    maxH = int(max(heightA, heightB))
    return order_points([[0, 0], [maxW, 0], [maxW, maxH], [0, maxH]])


def scan(image_true: np.ndarray, trained_model: torch.nn.Module, image_size: int = 384, BUFFER: int = 10):
    global preprocess_transforms

    IMAGE_SIZE = image_size
    half = IMAGE_SIZE // 2

    imH, imW, C = image_true.shape

    # Resize ảnh đầu vào cho model
    image_model = cv2.resize(image_true, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    scale_x, scale_y = imW / IMAGE_SIZE, imH / IMAGE_SIZE

    # Tiền xử lý & thêm batch dim
    image_model = preprocess_transforms(image_model)
    image_model = torch.unsqueeze(image_model, 0)

    # 👉 đảm bảo input tensor cùng device với model
    device = next(trained_model.parameters()).device
    image_model = image_model.to(device)

    with torch.no_grad():
        out = trained_model(image_model)["out"].cpu()

    # Giải phóng
    del image_model
    gc.collect()

    out = (
        torch.argmax(out, 1, keepdim=True)
        .permute(0, 2, 3, 1)[0]
        .numpy()
        .squeeze()
        .astype(np.int32)
    )

    r_H, r_W = out.shape
    _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
    _out_extended[half : half + IMAGE_SIZE, half : half + IMAGE_SIZE] = out * 255
    out = _out_extended.copy()
    del _out_extended; gc.collect()

    # Tìm contour lớn nhất
    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)
    corners = np.concatenate(corners).astype(np.float32)

    # Scale corner về toạ độ ảnh gốc
    corners[:, 0] = (corners[:, 0] - half) * scale_x
    corners[:, 1] = (corners[:, 1] - half) * scale_y

    # Nếu box vượt ngoài ảnh, mở rộng canvas & dịch góc
    if not (
        np.all(corners.min(0) >= (0, 0)) and np.all(corners.max(0) <= (imW, imH))
    ):
        left_pad = top_pad = right_pad = bottom_pad = 0
        rect = cv2.minAreaRect(corners.reshape(-1, 1, 2))
        box = np.int32(cv2.boxPoints(rect))

        if (dx := box[:, 0].min()) <= 0:
            left_pad = abs(dx) + BUFFER
        if (dx := box[:, 0].max()) >= imW:
            right_pad = dx - imW + BUFFER
        if (dy := box[:, 1].min()) <= 0:
            top_pad = abs(dy) + BUFFER
        if (dy := box[:, 1].max()) >= imH:
            bottom_pad = dy - imH + BUFFER

        image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, C), dtype=image_true.dtype)
        image_extended[top_pad : top_pad + imH, left_pad : left_pad + imW] = image_true

        image_true = image_extended.astype(np.float32)
        corners = box + np.array([left_pad, top_pad])

    corners = order_points(sorted(corners.tolist()))
    destination_corners = find_dest(corners)

    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    final = cv2.warpPerspective(
        image_true,
        M,
        (destination_corners[2][0], destination_corners[2][1]),
        flags=cv2.INTER_LANCZOS4,
    )

    return np.clip(final, 0, 255).astype(np.uint8)


# Link tải ảnh kết quả

def get_image_download_link(img: Image.Image, filename: str, text: str) -> str:
    buff = io.BytesIO()
    img.save(buff, format="JPEG")
    b64 = base64.b64encode(buff.getvalue()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'


STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / "static"
DOWNLOADS_PATH = STREAMLIT_STATIC_PATH / "downloads"
DOWNLOADS_PATH.mkdir(exist_ok=True)

IMAGE_SIZE = 384
preprocess_transforms = image_preprocess_transforms()

st.title("Document Scanner: Semantic Segmentation using DeepLabV3-PyTorch")

uploaded_file = st.file_uploader("Upload Document Image :", type=["png", "jpg", "jpeg"])
method = st.radio(
    "Select Document Segmentation Model:",
    ("MobilenetV3-Large", "Resnet-50"),
    horizontal=True,
)

col1, col2 = st.columns((6, 5))

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    model_name = "mbv3" if method == "MobilenetV3-Large" else "r50"
    model = load_model(model_name=model_name)  # tự chọn device

    with col1:
        st.subheader("Input")
        st.image(image, channels="BGR", use_column_width=True)

    with col2:
        st.subheader("Scanned")
        final = scan(image_true=image, trained_model=model, image_size=IMAGE_SIZE)
        st.image(final, channels="BGR", use_column_width=True)

    result = Image.fromarray(final[:, :, ::-1])  # chuyển từ BGR → RGB
    st.markdown(
        get_image_download_link(result, "output.png", "📥 Download Output"),
        unsafe_allow_html=True,
    )