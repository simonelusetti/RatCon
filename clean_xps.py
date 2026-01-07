import os
import shutil
import json
from PIL import Image, ImageDraw, ImageFont

# CHANGE THIS to the folder you want to scan
BASE_FOLDER = r"outputs/xps/"

for item in os.listdir(BASE_FOLDER):
    subfolder_path = os.path.join(BASE_FOLDER, item)

    if os.path.isdir(subfolder_path):
        has_png = any(
            file.lower().endswith(".png")
            for file in os.listdir(subfolder_path)
            if os.path.isfile(os.path.join(subfolder_path, file))
        )

        if not has_png:
            print(f"Deleting: {subfolder_path}")
            shutil.rmtree(subfolder_path)

FONT_SIZE = 28
PADDING = 10
BG_COLOR = (255, 255, 255)  # white
TEXT_COLOR = (0, 0, 0)      # black

def find_file(folder, ext):
    for f in os.listdir(folder):
        if f.lower().endswith(ext):
            return os.path.join(folder, f)
    return None

def get_system_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    # Fallback (small, but guaranteed)
    return ImageFont.load_default()

for sub in os.listdir(BASE_FOLDER):
    sub_path = os.path.join(BASE_FOLDER, sub)
    if not os.path.isdir(sub_path):
        continue

    png_path = find_file(sub_path, ".png")
    argv_path = find_file(sub_path, ".argv.json")

    if not png_path or not argv_path:
        continue

    # Load args
    with open(argv_path, "r") as f:
        args = json.load(f)

    text = " | ".join(args)

    # Open image
    img = Image.open(png_path).convert("RGB")
    width, height = img.size

    draw = ImageDraw.Draw(img)

    font = get_system_font(FONT_SIZE)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    new_height = height + text_height + 2 * PADDING
    new_img = Image.new("RGB", (width, new_height), BG_COLOR)

    # Paste original image lower
    new_img.paste(img, (0, text_height + 2 * PADDING))

    draw = ImageDraw.Draw(new_img)
    draw.text(
        (PADDING, PADDING),
        text,
        fill=TEXT_COLOR,
        font=font
    )

    new_img.save(png_path)
    print(f"Annotated: {png_path}")
