#!/usr/bin/env python3
import os, time, random, configparser
from itertools import groupby

import cv2
import numpy as np
from PIL import Image
from IT8951.display import AutoEPDDisplay
from IT8951 import constants

BASE_DIR    = os.path.dirname(__file__)
CONFIG_FILE = os.path.join(BASE_DIR, 'config.txt')
ACTIVE_DIR  = os.path.join(BASE_DIR, 'active-frame')
VIDEO_PATH  = os.path.join(BASE_DIR, 'big-trouble-little-china.mp4')

def read_config():
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',';'))
    cfg.read(CONFIG_FILE)
    return (
        cfg.getint('General','display_time',   fallback=300),
        cfg.getint('General','frame_interval', fallback=30),
        cfg.getint('General','jitter',         fallback=5),
        cfg.getint('General','fat_bits',       fallback=1),
        cfg.get('General','update_mode',       fallback='DU').upper(),
        cfg.getint('General','cluster_size',   fallback=512),
        cfg.getfloat('General','chunk_pause',  fallback=0.05)
    )

def ensure_active_dir():
    if os.path.isdir(ACTIVE_DIR):
        for f in os.listdir(ACTIVE_DIR):
            os.remove(os.path.join(ACTIVE_DIR, f))
    else:
        os.makedirs(ACTIVE_DIR)

def atkinson_dither(im):
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape
    for y in range(h):
        for x in range(w):
            old = arr[y, x]
            new = 255 if old > 128 else 0
            arr[y, x] = new
            err = old - new
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
        ret, frame = cap.read()
        t = 0.0
    return frame, t

def prepare_frame(cap, current_t, duration,
                  disp_w, disp_h,
                  fat_bits, frame_interval, jitter, cluster_size):
    # 1) sample
    next_t = current_t + frame_interval + random.uniform(-jitter, jitter)
    frame, sampled_t = sample_frame(cap, next_t, duration)

    # 2) load grayscale & letterbox
    ensure_active_dir()
    tmp = os.path.join(ACTIVE_DIR,'frame.jpg')
    cv2.imwrite(tmp, frame)
    im = Image.open(tmp).convert('L')
    iw, ih = im.size

    scale = min(disp_w/iw, disp_h/ih)
    nw = int(iw*scale) - (int(iw*scale)%fat_bits)
    nh = int(ih*scale) - (int(ih*scale)%fat_bits)
    imr = im.resize((nw,nh), Image.Resampling.LANCZOS)
    x_off = (disp_w-nw)//2
    y_off = (disp_h-nh)//2

    letterbox = Image.new('L',(disp_w,disp_h),255)
    letterbox.paste(0,(0,0,disp_w,y_off))
    letterbox.paste(0,(0,y_off+nh,disp_w,disp_h))

    # 3) build full offscreen image (letterbox + dither)
    fb_off = letterbox.copy()
    small = imr.resize((nw//fat_bits, nh//fat_bits), Image.Resampling.LANCZOS)
    dsmall = atkinson_dither(small)
    dfull  = dsmall.resize((nw,nh), Image.Resampling.NEAREST)
    fb_off.paste(dfull,(x_off,y_off))

    # 4) compute smoke-order
    gray = np.array(small)
    coords = [(x,y,gray[y,x])
              for y in range(nh//fat_bits)
              for x in range(nw//fat_bits)
              if dsmall.getpixel((x,y))==0]
    coords.sort(key=lambda e:e[2])
    rnd=[]
    for _,grp in groupby(coords,key=lambda e:e[2]):
        bucket=list(grp); random.shuffle(bucket); rnd+=bucket

    # 5) split into chunks of cluster_size
    chunks=[]
    for i in range(0,len(rnd),cluster_size):
        chunk = rnd[i:i+cluster_size]
        # convert to display coords
        pts=[(x_off + x*fat_bits, y_off + y*fat_bits) for x,y,_ in chunk]
        chunks.append(pts)

    return letterbox, fb_off, chunks, sampled_t

def main():
    display_time, frame_interval, jitter, fat_bits, update_mode, cluster_size, chunk_pause = read_config()
    mode_val = constants.DisplayModes.DU  # we use DU for multi-pass

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {VIDEO_PATH}")
    fps      = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps

    epd = AutoEPDDisplay(vcom=-1.80)
    epd.clear()
    disp_w, disp_h = epd.width, epd.height

    t=0.0
    while True:
        # 1) prepare everything off-screen
        letterbox, fb_off, chunks, t = prepare_frame(
            cap, t, duration,
            disp_w, disp_h,
            fat_bits, frame_interval, jitter, cluster_size
        )

        # 2) draw the static letterbox background
        epd.frame_buf = letterbox
        epd.draw_full(constants.DisplayModes.GC16)

        # 3) reveal in multi-pass
        for pts in chunks:
            for x0,y0 in pts:
                # paste one fat_bits block
                block = fb_off.crop((x0,y0,x0+fat_bits,y0+fat_bits))
                epd.frame_buf.paste(block,(x0,y0))
            epd.draw_partial(1)                 # DU full‐frame refresh
            time.sleep(chunk_pause)

        # 4) hold
        time.sleep(display_time)

if __name__=='__main__':
    main()