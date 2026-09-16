import sys
import os
import io
from pathlib import Path
from PIL import Image
import torch

print("=" * 60)
print(" RUNNING PRODUCTION READINESS & NEURAL STYLE TRANSFER TESTS")
print("=" * 60)

# 1. Test imports from current project (verifying Gunicorn entrypoint 'app:app')
try:
    from app import app, encoder, decoder, device, style_transfer, save_image, validate_image_file, resolve_image_path, PROJECT_ROOT, UPLOAD_FOLDER, EXAMPLES_FOLDER
    print("[PASS] Gunicorn entrypoint verified: 'app:app' imported successfully.")
except Exception as e:
    print(f"[FAIL] Error importing modules: {e}")
    sys.exit(1)

# 2. Test device and models
print(f"Active Device: {device}")
assert not encoder.training, "[FAIL] Encoder should be in eval mode"
assert not decoder.training, "[FAIL] Decoder should be in eval mode"
print("[PASS] Encoder and Decoder are loaded and in evaluation mode.")

# 3. Test image loading
content_path = EXAMPLES_FOLDER / "brad_pitt.jpg"
style_path = EXAMPLES_FOLDER / "sketch.png"

assert content_path.is_file(), f"Missing test content image: {content_path}"
assert style_path.is_file(), f"Missing test style image: {style_path}"
print(f"[PASS] Example images verified: {content_path.name}, {style_path.name}")

# 4. Test Style Transfer with Alpha = 0.0, 0.5, 1.0
content_img = Image.open(content_path).convert('RGB')
style_img = Image.open(style_path).convert('RGB')

for test_alpha in [0.0, 0.5, 1.0]:
    print(f"Testing inference with alpha={test_alpha}...")
    with torch.no_grad():
        stylized_tensor = style_transfer(content_img, style_img, encoder, decoder, test_alpha, device)
    
    assert stylized_tensor is not None, "Stylized tensor is None"
    assert stylized_tensor.dim() == 4, f"Expected 4D tensor, got shape {stylized_tensor.shape}"
    assert stylized_tensor.shape[1] == 3, f"Expected 3 channels, got shape {stylized_tensor.shape}"
    
    out_name = f"test_out_alpha_{str(test_alpha).replace('.', '_')}.jpg"
    out_path = UPLOAD_FOLDER / out_name
    save_image(stylized_tensor, str(out_path))
    assert out_path.is_file(), f"Output image was not saved: {out_path}"
    assert out_path.stat().st_size > 0, "Output image file is empty"
    
    # Verify saved image is readable
    with Image.open(out_path) as saved_img:
        saved_img.verify()
    print(f"[PASS] Alpha={test_alpha} successfully generated and verified ({out_path.name}, {out_path.stat().st_size} bytes)")
    
    # Clean up test output
    out_path.unlink()

# 5. Test Image Validation (corrupt file handling)
dummy_bad_path = UPLOAD_FOLDER / "corrupt_test.jpg"
dummy_bad_path.write_text("NOT_AN_IMAGE_CONTENT", encoding="utf-8")
assert not validate_image_file(dummy_bad_path), "Failed to detect corrupt image file"
dummy_bad_path.unlink()
print("[PASS] Corrupt image validation safety check passed.")

# 6. Test Flask Routes via Test Client
client = app.test_client()

# GET /health
response_health = client.get('/health')
assert response_health.status_code == 200, f"GET /health returned {response_health.status_code}"
assert response_health.json == {"status": "ok"}, f"Unexpected health json: {response_health.json}"
print("[PASS] Flask GET /health returned HTTP 200 with {'status': 'ok'}.")

# GET /
response = client.get('/')
assert response.status_code == 200, f"GET / returned {response.status_code}"
assert b"STYLEFORGE AI" in response.data, "App title not in response"
assert b"Active Inference" in response.data, "Device indicator not in response"
print("[PASS] Flask GET / returned HTTP 200 with clean page.")

# GET /examples/sketch.png
response_ex = client.get('/examples/sketch.png')
assert response_ex.status_code == 200, f"GET /examples/sketch.png returned {response_ex.status_code}"
print("[PASS] Flask GET /examples/sketch.png returned HTTP 200.")

# POST with selected example content and style
post_data = {
    'content_path': 'examples/brad_pitt.jpg',
    'style_path': 'examples/sketch.png',
    'alpha': 0.8
}
response_post = client.post('/', data=post_data)
assert response_post.status_code == 200, f"POST / returned {response_post.status_code}"
assert b"Stylized Result" in response_post.data, "Stylized result section not in response after style transfer"
print("[PASS] Flask POST / with example images successfully generated stylized result!")

print("=" * 60)
print(" ALL PRODUCTION & STYLE TRANSFER TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
