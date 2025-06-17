#!/usr/bin/env python3
import os, time, random, configparser
from itertools import groupby

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageEnhance
from IT8951.display import AutoEPDDisplay
from IT8951 import constants

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(__file__)
CONFIG_FILE    = os.path.join(BASE_DIR, 'config.txt')
ACTIVE_DIR     = os.path.join(BASE_DIR, 'active-frame')
VIDEO_PATH     = os.path.join(BASE_DIR, 'big-trouble-little-china.mp4')

# ←─ your person/bg model here
SEG_MODEL_PATH = os.path.join(BASE_DIR,
    'deeplabv3_257_mv2_1.0_257.tflite')

# ─── Config Reader ─────────────────────────────────────────────────────────────
def read_config():
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#",";"))
    cfg.read(CONFIG_FILE)
    return (
        cfg.getfloat('General','start_time',    fallback=0.0),
        cfg.getint  ('General','display_time',  fallback=300),
        cfg.getint  ('General','frame_interval',fallback=30),
        cfg.getint  ('General','jitter',        fallback=5),
        cfg.getint  ('General','fat_bits',      fallback=1),
        cfg.get    ('General','update_mode',    fallback='DU').upper(),
        cfg.getint  ('General','cluster_size',  fallback=512),
        cfg.getfloat('General','chunk_pause',   fallback=0.05),
        cfg.getfloat('General','brightness',    fallback=1.0),
        cfg.getfloat('General','contrast',      fallback=1.0),
    )

# ─── Utilities ────────────────────────────────────────────────────────────────
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

# ─── Segmentation ─────────────────────────────────────────────────────────────
def load_segmentation_model():
    if os.path.isfile(SEG_MODEL_PATH):
        interp = tflite.Interpreter(model_path=SEG_MODEL_PATH)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        print("🧩 TF-Lite PERSON/BG model loaded ✔️")
        print(f"   • input  shape={inp['shape']}, dtype={inp['dtype']}")
        print(f"   • output shape={out['shape']}, dtype={out['dtype']}")
        return interp, inp, out
    print("⚠️  No segmentation model found, skipping silhouette")
    return None, None, None

def get_silhouette_mask(sess, im_color, target_size, _):
    interp, inp, out = sess

    # 1) Resize + BGR→RGB
    _, h_in, w_in, _ = inp['shape']
    img = cv2.resize(im_color, (w_in, h_in))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2) Prepare tensor
    if inp['dtype'] == np.uint8:
        scale, zp = inp['quantization']
        tensor = ((img/scale) + zp).astype(np.uint8)
    else:
        # float32: normalize to [0,1]
        tensor = (img.astype(np.float32) / 255.0)

    interp.set_tensor(inp['index'], np.expand_dims(tensor,0))

    # 3) Infer
    interp.invoke()
    raw = interp.get_tensor(out['index'])[0]  # H×W×C

    C = raw.shape[-1]
    if C == 2:
        mask = (raw[:,:,1] > raw[:,:,0]).astype(np.uint8)
    else:
        cls = np.argmax(raw, axis=-1)
        mask = (cls == 15).astype(np.uint8)

    # 4) Downsample
    small = cv2.resize(mask, target_size,
                       interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(ACTIVE_DIR,'small-mask.png'),
                small*255)
    return small

# ─── Frame Sampling ────────────────────────────────────────────────────────────
def sample_frame(cap, t, duration):
    if t >= duration: t %= duration
    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_MSEC,0)
        ret, frame = cap.read()
        t = 0.0
    return frame, t

# ─── Frame Prep ────────────────────────────────────────────────────────────────
def prepare_frame(cap, current_t, duration, disp_w, disp_h,
                  fat_bits, frame_interval, jitter, cluster_size,
                  brightness, contrast, seg_sess):

    t_next = current_t + frame_interval + random.uniform(-jitter, jitter)
    frame_color, sampled_t = sample_frame(cap, t_next, duration)
    print(f"🖼️ Frame sampled at t={sampled_t:.2f}")

    ensure_active_dir()
    raw_path = os.path.join(ACTIVE_DIR,'frame.jpg')
    cv2.imwrite(raw_path, frame_color)
    print("💾 Saved raw frame")

    im_gray = Image.open(raw_path).convert('L')
    if brightness!=1.0:
        im_gray = ImageEnhance.Brightness(im_gray).enhance(brightness)
        print(f"⚙️ Brightness: {brightness}")
    if contrast!=1.0:
        im_gray = ImageEnhance.Contrast(im_gray).enhance(contrast)
        print(f"🎛️ Contrast: {contrast}")

    iw, ih = im_gray.size
    scale = min(disp_w/iw, disp_h/ih)
    nw = (int(iw*scale)//fat_bits)*fat_bits
    nh = (int(ih*scale)//fat_bits)*fat_bits
    im_gray_r  = im_gray.resize((nw,nh), Image.Resampling.LANCZOS)
    im_color_r = cv2.resize(frame_color,(nw,nh),
                            interpolation=cv2.INTER_LANCZOS4)
    print(f"↔️ Resized to {nw}×{nh}")
    x_off,y_off = (disp_w-nw)//2,(disp_h-nh)//2

    interp, inp, out = seg_sess
    if interp:
        small_mask = get_silhouette_mask(
            seg_sess, im_color_r,
            (nw//fat_bits, nh//fat_bits), im_gray_r
        )
    else:
        small_mask = np.zeros((nh//fat_bits, nw//fat_bits),dtype=np.uint8)
    print("🔍 Silhouette mask computed")

    letterbox = Image.new('L',(disp_w,disp_h),255)
    letterbox.paste(0,(0,0,disp_w,y_off))
    letterbox.paste(0,(0,y_off+nh,disp_w,disp_h))

    fb_off = letterbox.copy()
    small   = im_gray_r.resize((nw//fat_bits, nh//fat_bits),
                               Image.Resampling.LANCZOS)
    dsmall  = atkinson_dither(small)
    dfull   = dsmall.resize((nw,nh),
                             Image.Resampling.NEAREST)
    fb_off.paste(dfull,(x_off,y_off))
    fb_off.save(os.path.join(ACTIVE_DIR,'dithered.png'))
    print("💾 Dither preview saved")

    coords_fig, coords_bg = [], []
    gray_arr = np.array(small)
    for y in range(nh//fat_bits):
        for x in range(nw//fat_bits):
            if dsmall.getpixel((x,y))==0:
                cell=(x,y,gray_arr[y,x])
                (coords_fig if small_mask[y,x] else coords_bg).append(cell)
    coords_fig.sort(key=lambda e:e[2])
    coords_bg .sort(key=lambda e:e[2])
    print(f"👤 Figure cells: {len(coords_fig)}, background: {len(coords_bg)}")

    rnd_fig, rnd_bg = [], []
    for _,g in groupby(coords_fig,key=lambda e:e[2]):
        lst=list(g); random.shuffle(lst); rnd_fig+=lst
    for _,g in groupby(coords_bg,key=lambda e:e[2]):
        lst=list(g); random.shuffle(lst); rnd_bg+=lst

    chunks=[]
    for seq in (rnd_fig, rnd_bg):
        for i in range(0,len(seq),cluster_size):
            pts=[(x_off+x*fat_bits, y_off+y*fat_bits)
                 for x,y,_ in seq[i:i+cluster_size]]
            chunks.append(pts)
    print(f"🔧 Total chunks: {len(chunks)} (fig then bg)")

    return letterbox, fb_off, chunks, sampled_t

def main():
    (start_t, display_time, frame_interval, jitter, fat_bits,
     update_mode, cluster_size, chunk_pause, brightness, contrast) = read_config()

    seg_sess = load_segmentation_model()

    cap      = cv2.VideoCapture(VIDEO_PATH)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps

    epd = AutoEPDDisplay(vcom=-1.80)
    epd.clear()
    disp_w,disp_h = epd.width,epd.height

    print(f"▶️ Beta 0.5 start={start_t}s, br={brightness}, cr={contrast}, fat={fat_bits}, cluster={cluster_size}")
    t = start_t

    while True:
        lb, fb, chunks, t = prepare_frame(
            cap,t,duration,
            disp_w,disp_h,
            fat_bits,frame_interval,jitter,cluster_size,
            brightness,contrast,seg_sess
        )

        epd.frame_buf  = lb
        epd.draw_full(constants.DisplayModes.GC16)
        print("🟢 Letterbox drawn")

        for pts in chunks:
            for x0,y0 in pts:
                blk = fb.crop((x0,y0,x0+fat_bits,y0+fat_bits))
                epd.frame_buf.paste(blk,(x0,y0))
            epd.draw_partial(1)
            time.sleep(chunk_pause)

        print(f"⏱️ Holding for {display_time}s")
        time.sleep(display_time)

if __name__=='__main__':
    main()