import os
import time
import uuid
import logging
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import NumberRange
from werkzeug.utils import secure_filename

# Import existing AdaIN models & utilities
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Dynamic Project Paths (Machine-Independent)
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_FOLDER = PROJECT_ROOT / "static" / "uploads"
EXAMPLES_FOLDER = PROJECT_ROOT / "examples"
VGG_PATH = PROJECT_ROOT / "vgg_normalised.pth"
DECODER_PATH = PROJECT_ROOT / "experiment" / "final_exp" / "decoder_final.pth"

# Ensure upload directory exists
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Flask Application Setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'nst-fallback-secret-key-36dd776c')
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
Bootstrap(app)

# Form Definition
class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField('Content Path')
    style_path = HiddenField('Style Path')
    alpha = FloatField('Alpha', default=1.0, validators=[
        NumberRange(min=0.0, max=1.0, message="Alpha must be between 0.0 and 1.0")
    ])
    submit = SubmitField('Transfer Style')

# Hardware Device Detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    device_info = f"CUDA ({gpu_name})"
    logger.info(f"Device: CUDA | GPU: {gpu_name}")
else:
    device_info = "CPU"
    logger.info("Device: CPU (CUDA is unavailable; running inference reliably on CPU)")

# Model Checkpoint Verification
if not VGG_PATH.is_file():
    raise FileNotFoundError(
        f"Missing VGG model checkpoint at: '{VGG_PATH}'. "
        "Please ensure 'vgg_normalised.pth' exists in the project root."
    )

if not DECODER_PATH.is_file():
    raise FileNotFoundError(
        f"Missing Decoder checkpoint at: '{DECODER_PATH}'. "
        "Please ensure 'decoder_final.pth' exists in 'experiment/final_exp/'."
    )

# Load Models
logger.info(f"Loading VGG encoder from {VGG_PATH.name}...")
encoder = VGGEncoder(str(VGG_PATH), map_location=device).to(device)

logger.info(f"Loading Decoder from {DECODER_PATH.name}...")
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(str(DECODER_PATH), map_location=device))

encoder.eval()
decoder.eval()
logger.info("Models loaded successfully and set to evaluation mode.")


def allowed_file(filename):
    """Check if the filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def validate_image_file(file_path):
    """Verify that the image file is valid and readable by PIL."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img.convert('RGB')
        return True
    except Exception as err:
        logger.warning(f"Corrupted or invalid image file: {file_path} - {err}")
        return False


def style_transfer(content_image, style_image, encoder_model, decoder_model, alpha_val, compute_device):
    """Execute AdaIN style transfer pipeline."""

    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    content_tensor = content_transform(content_image).unsqueeze(0).to(compute_device)
    style_tensor = style_transform(style_image).unsqueeze(0).to(compute_device)

    logger.info(
        f"Input tensors created: content={tuple(content_tensor.shape)}, "
        f"style={tuple(style_tensor.shape)}"
    )

    with torch.no_grad():

        logger.info("Starting content VGG encoding...")
        content_feats = encoder_model(content_tensor, is_test=True)
        logger.info(f"Content VGG encoding completed: {tuple(content_feats.shape)}")

        logger.info("Starting style VGG encoding...")
        style_feats = encoder_model(style_tensor, is_test=True)
        logger.info(f"Style VGG encoding completed: {tuple(style_feats.shape)}")

        logger.info("Starting AdaIN...")
        stylized_feats = adaptive_instance_normalization(
            content_feats,
            style_feats
        )
        logger.info("AdaIN completed")

        stylized_feats = (
            alpha_val * stylized_feats
            + (1.0 - alpha_val) * content_feats
        )

        logger.info("Starting decoder...")
        stylized_image = decoder_model(stylized_feats)
        logger.info("Decoder completed")

    return stylized_image


def save_image(image_tensor, save_path):
    """Safely save a PyTorch image tensor to disk as an image file."""
    img = image_tensor.cpu().clone().squeeze(0).clamp(0, 1)
    pil_img = transforms.ToPILImage()(img)
    pil_img.save(save_path)


def resolve_image_path(filename_or_relpath):
    """Resolve a relative filename or example reference to a real filesystem path."""
    if not filename_or_relpath:
        return None
    # Check if it refers to an example
    if filename_or_relpath.startswith("examples/"):
        rel_name = filename_or_relpath.split("examples/", 1)[1]
        target = EXAMPLES_FOLDER / rel_name
        if target.is_file():
            return target
    # Check in uploads
    target = UPLOAD_FOLDER / filename_or_relpath
    if target.is_file():
        return target
    # Check in examples directly
    target = EXAMPLES_FOLDER / filename_or_relpath
    if target.is_file():
        return target
    return None


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == 'POST':
        # 1. Handle Content Image
        if form.content.data and hasattr(form.content.data, 'filename') and form.content.data.filename:
            raw_filename = form.content.data.filename
            if allowed_file(raw_filename):
                safe_name = secure_filename(raw_filename)
                unique_name = f"content_{int(time.time())}_{uuid.uuid4().hex[:6]}_{safe_name}"
                dest_path = UPLOAD_FOLDER / unique_name
                form.content.data.save(str(dest_path))

                if validate_image_file(dest_path):
                    content_filename = unique_name
                    form.content_path.data = unique_name
                else:
                    if dest_path.is_file():
                        dest_path.unlink()
                    error = "Uploaded Content Image is invalid or corrupted."
            else:
                error = f"Invalid Content Image extension. Allowed formats: {', '.join(app.config['ALLOWED_EXTENSIONS'])}"
        elif form.content_path.data:
            content_filename = form.content_path.data

        # 2. Handle Style Image
        if not error:
            if form.style.data and hasattr(form.style.data, 'filename') and form.style.data.filename:
                raw_filename = form.style.data.filename
                if allowed_file(raw_filename):
                    safe_name = secure_filename(raw_filename)
                    unique_name = f"style_{int(time.time())}_{uuid.uuid4().hex[:6]}_{safe_name}"
                    dest_path = UPLOAD_FOLDER / unique_name
                    form.style.data.save(str(dest_path))

                    if validate_image_file(dest_path):
                        style_filename = unique_name
                        form.style_path.data = unique_name
                    else:
                        if dest_path.is_file():
                            dest_path.unlink()
                        error = "Uploaded Style Image is invalid or corrupted."
                else:
                    error = f"Invalid Style Image extension. Allowed formats: {', '.join(app.config['ALLOWED_EXTENSIONS'])}"
            elif form.style_path.data:
                style_filename = form.style_path.data

        # 3. Check for required images
        if not error:
            if not content_filename:
                error = "Please upload or select a Content Image."
            elif not style_filename:
                error = "Please upload or select a Style Image."

        # 4. Perform Style Transfer
        if not error and content_filename and style_filename:
            content_real_path = resolve_image_path(content_filename)
            style_real_path = resolve_image_path(style_filename)

            if not content_real_path or not content_real_path.is_file():
                error = "Content image file could not be found on server."
            elif not style_real_path or not style_real_path.is_file():
                error = "Style image file could not be found on server."
            else:
                try:
                    # Validate alpha parameter
                    try:
                        alpha_val = float(form.alpha.data) if form.alpha.data is not None else 1.0
                    except (TypeError, ValueError):
                        alpha_val = 1.0
                    alpha_val = max(0.0, min(1.0, alpha_val))

                    logger.info(f"Running style transfer: content={content_real_path.name}, style={style_real_path.name}, alpha={alpha_val}")

                    content_img = Image.open(content_real_path).convert('RGB')
                    style_img = Image.open(style_real_path).convert('RGB')

                    stylized_tensor = style_transfer(content_img, style_img, encoder, decoder, alpha_val, device)

                    result_filename = f"stylized_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
                    result_path = UPLOAD_FOLDER / result_filename
                    save_image(stylized_tensor, str(result_path))

                    result_image = result_filename
                    logger.info(f"Generated result saved successfully: {result_filename}")
                except Exception as ex:
                    logger.error(f"Inference error during style transfer: {ex}", exc_info=True)
                    error = f"An error occurred during neural style transfer: {str(ex)}"

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
        device_info=device_info
    )


@app.route('/uploads/<path:filename>')
def send_image(filename):
    """Serve uploaded and generated result images."""
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    """Serve included example images."""
    return send_from_directory(str(EXAMPLES_FOLDER), filename)


@app.route('/health', methods=['GET'])
def health():
    """Lightweight health check endpoint for monitoring, Docker, and Spaces."""
    return {"status": "ok"}, 200


@app.errorhandler(413)
def request_entity_too_large(e):
    form = UploadForm()
    return render_template(
        'index.html',
        form=form,
        result_image=None,
        content_image=None,
        style_image=None,
        error="File size is too large! Maximum allowed upload size is 16MB.",
        device_info=device_info
    ), 413


if __name__ == '__main__':
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 7860))
    print("=" * 60)
    print(" AdaIN Neural Style Transfer Web Application")
    print(f" Local URL:   http://{host}:{port}")
    print(f" Compute:     {device_info}")
    print("=" * 60)
    app.run(host=host, port=port, debug=False)
