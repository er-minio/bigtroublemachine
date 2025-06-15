#!/usr/bin/env python3
import os, time, random, configparser
from itertools import groupby

import cv2, numpy as np
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
        cfg.getint('General','display_time',     fallback=300),
        cfg.getint('General','frame_interval',   fallback=30),
        cfg.getint('General','jitter',           fallback=5),
        cfg.getint('General','fat_bits',         fallback=1),
        cfg.get('General','update_mode',         fallback='A2').upper(),
        cfg.getint('General','window_multiplier',fallback=2),
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
    if t >= duration: t %= duration
    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_MSEC,0)
        ret, frame = cap.read()
        t=0
    return frame, t

def main():
    # load config
    display_time, frame_interval, jitter, fat_bits, update_mode, window_multiplier = read_config()
    mode_val = constants.DisplayModes.DU if update_mode=='DU' else constants.DisplayModes.A2
    print(f"➡️ fat_bits = {fat_bits}, window_multiplier = {window_multiplier}")
    print(f"🕑 frame persistance time = {display_time} seconds")

    # open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): raise IOError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps

    # init display
    display = AutoEPDDisplay(vcom=-1.80)
    display.clear()
    disp_w, disp_h = display.width, display.height

    t = 0.0
    while True:
        # sample frame
        t_next = t + frame_interval + random.uniform(-jitter,jitter)
        frame, t = sample_frame(cap, t_next, duration)

        # save and load
        ensure_active_dir()
        cv2.imwrite(os.path.join(ACTIVE_DIR,'frame.jpg'), frame)
        print("🟢 Frame selected")
        im = Image.open(os.path.join(ACTIVE_DIR,'frame.jpg')).convert('L')

        # letterbox
        iw, ih = im.size
        scale = min(disp_w/iw, disp_h/ih)
        nw = int(iw*scale) - (int(iw*scale)%fat_bits)
        nh = int(ih*scale) - (int(ih*scale)%fat_bits)
        imr = im.resize((nw,nh), Image.Resampling.LANCZOS)
        x0_off = (disp_w-nw)//2; y0_off = (disp_h-nh)//2

        # draw bars
        bg = Image.new('L',(disp_w,disp_h),255)
        bg.paste(0,(0,0,disp_w,y0_off))
        bg.paste(0,(0,y0_off+nh,disp_w,disp_h))
        display.frame_buf = bg
        display.draw_full(constants.DisplayModes.GC16)
        print("🟢 Letterboxed")

        # set mode
        display.draw_full(mode_val)
        print(f"🟢 Draw mode = {update_mode}")

        # dither
        sw, sh = nw//fat_bits, nh//fat_bits
        small = imr.resize((sw,sh), Image.Resampling.LANCZOS)
        print(f"➡️ image size = {sw}×{sh}")
        dsmall = atkinson_dither(small)
        gray  = np.array(small)

        # build full dither
        dfull = dsmall.resize((nw,nh), Image.Resampling.NEAREST)
        full  = Image.new('L',(disp_w,disp_h),0)
        full.paste(dfull,(x0_off,y0_off))
        full.save(os.path.join(ACTIVE_DIR,'dithered.png'))
        print("🟢 Dither saved")

        # reveal
        coords = [(x,y,gray[y,x]) for y in range(sh) for x in range(sw) if dsmall.getpixel((x,y))==0]
        coords.sort(key=lambda e:e[2])
        rnd = []
        for v,grp in groupby(coords,key=lambda e:e[2]):
            b=list(grp); random.shuffle(b); rnd+=b

        # batch windows
        win = fat_bits*window_multiplier
        seen=set(); wins=[]
        for xs,ys,_ in rnd:
            wx,wy = xs//window_multiplier, ys//window_multiplier
            if (wx,wy) not in seen:
                seen.add((wx,wy)); wins.append((wx,wy))

        # update each
        for wx,wy in wins:
            px = x0_off + wx*win; py = y0_off + wy*win
            blk = full.crop((px,py,px+win,py+win))
            display.frame_buf.paste(blk,(px,py))
            display.update(blk.tobytes(), (px,py), (win,win), mode_val)

        print("✅ Done")
        time.sleep(display_time)

if __name__=='__main__':
    main()