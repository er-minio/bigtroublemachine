#!/usr/bin/env python3
import os
import time
import random
import configparser
from itertools import groupby

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from IT8951.display import AutoEPDDisplay
from IT8951 import constants

BASE_DIR    = os.path.dirname(__file__)
CONFIG_FILE = os.path.join(BASE_DIR, 'config.txt')
ACTIVE_DIR  = os.path.join(BASE_DIR, 'active-frame')
VIDEO_PATH  = os.path.join(BASE_DIR, 'big-trouble-little-china.mp4')

# Beta 0.1: multi-pass DU reveal with adjustable brightness and concise debug

def read_config():
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#",";"))
    cfg.read(CONFIG_FILE)
    return (
        cfg.getint('General','display_time',   fallback=300),
        cfg.getint('General','frame_interval', fallback=30),
        cfg.getint('General','jitter',         fallback=5),
        cfg.getint('General','fat_bits',       fallback=1),
        cfg.get('General','update_mode',       fallback='DU').upper(),
        cfg.getint('General','cluster_size',   fallback=512),
        cfg.getfloat('General','chunk_pause',  fallback=0.05),
        cfg.getfloat('General','brightness',   fallback=1.0)
    )


def ensure_active_dir():
    if os.path.isdir(ACTIVE_DIR):
        for f in os.listdir(ACTIVE_DIR):
            os.remove(os.path.join(ACTIVE_DIR, f))
    else:
        os.makedirs(ACTIVE_DIR)


def atkinson_dither(im: Image.Image) -> Image.Image:
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape
    for y in range(h):
        for x in range(w):
            old = arr[y, x]; new = 255 if old > 128 else 0
            arr[y, x] = new; err = old - new
            for dx, dy in [(1,0),(2,0),(-1,1),(0,1),(1,1),(0,2)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h:
                    arr[ny, nx] += err/8
    return Image.fromarray(np.clip(arr,0,255).astype(np.uint8))


def sample_frame(cap, t, duration):
    if t >= duration:
        t %= duration
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        ret, frame = cap.read(); t = 0.0
    return frame, t


def prepare_frame(cap, current_t, duration,
                  disp_w, disp_h,
                  fat_bits, frame_interval, jitter, cluster_size,
                  brightness):
    # 1) sample next frame
    next_t = current_t + frame_interval + random.uniform(-jitter, jitter)
    frame, sampled_t = sample_frame(cap, next_t, duration)
    print(f"🖼️ Frame sampled at t={sampled_t:.2f}")

    # 2) save & open grayscale, apply brightness
    ensure_active_dir()
    raw_path = os.path.join(ACTIVE_DIR, 'frame.jpg')
    cv2.imwrite(raw_path, frame)
    print(f"💾 Saved raw frame to {os.path.basename(raw_path)}")
    im = Image.open(raw_path).convert('L')
    if brightness != 1.0:
        im = ImageEnhance.Brightness(im).enhance(brightness)
        print(f"⚙️ Brightness adjusted: {brightness}")
    iw, ih = im.size

    # 3) resize & aligned
    scale = min(disp_w/iw, disp_h/ih)
    nw = (int(iw*scale)//fat_bits)*fat_bits
    nh = (int(ih*scale)//fat_bits)*fat_bits
    imr = im.resize((nw, nh), Image.Resampling.LANCZOS)
    print(f"↔️ Resized to {nw}×{nh}")
    x_off, y_off = (disp_w-nw)//2, (disp_h-nh)//2

    # 4) build letterbox background
    letterbox = Image.new('L', (disp_w, disp_h), 255)
    letterbox.paste(0, (0, 0, disp_w, y_off))
    letterbox.paste(0, (0, y_off+nh, disp_w, disp_h))
    print("📐 Letterbox created")

    # 5) build off-screen full image (letterbox + dithered content)
    fb_off = letterbox.copy()
    small = imr.resize((nw//fat_bits, nh//fat_bits), Image.Resampling.LANCZOS)
    dsmall = atkinson_dither(small)
    dfull = dsmall.resize((nw, nh), Image.Resampling.NEAREST)
    fb_off.paste(dfull, (x_off, y_off))
    dither_path = os.path.join(ACTIVE_DIR, 'dithered.png')
    fb_off.save(dither_path)
    print(f"💾 Saved dither preview to {os.path.basename(dither_path)}")

    # 6) compute "smoke" order of dither pixels
    gray = np.array(small)
    coords = [(x, y, gray[y, x])
              for y in range(nh//fat_bits) for x in range(nw//fat_bits)
              if dsmall.getpixel((x, y)) == 0]
    coords.sort(key=lambda e: e[2])
    rnd = []
    for _, grp in groupby(coords, key=lambda e: e[2]):
        bucket = list(grp); random.shuffle(bucket); rnd.extend(bucket)
    print(f"🔢 Total dark cells: {len(rnd)}")

    # 7) split into chunks
    chunks = []
    for i in range(0, len(rnd), cluster_size):
        pts = [(x_off + x*fat_bits, y_off + y*fat_bits) for x,y,_ in rnd[i:i+cluster_size]]
        chunks.append(pts)
    print(f"🔧 Split into {len(chunks)} chunks (size={cluster_size})")

    return letterbox, fb_off, chunks, sampled_t


def main():
    dt, fi, jit, fat, mode, cs, cp, br = read_config()
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps

    epd = AutoEPDDisplay(vcom=-1.80)
    epd.clear(); dw, dh = epd.width, epd.height

    print(f"▶️ Starting: brightness={br}, fat_bits={fat}, cluster={cs}")
    t = 0.0
    while True:
        letterbox, fb_off, chunks, t = prepare_frame(
            cap, t, dur, dw, dh, fat, fi, jit, cs, br
        )

        epd.frame_buf = letterbox
        epd.draw_full(constants.DisplayModes.GC16)

        for pts in chunks:
            for x0, y0 in pts:
                epd.frame_buf.paste(fb_off.crop((x0,y0,x0+fat,y0+fat)), (x0,y0))
            epd.draw_partial(1)
            time.sleep(cp)

        time.sleep(dt)

if __name__ == '__main__':
    main()