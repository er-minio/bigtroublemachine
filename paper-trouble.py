#!/usr/bin/env python3
import os, time, random, configparser, shutil, json
from itertools import groupby
from datetime import datetime

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
GIF_DIR        = os.path.join(BASE_DIR, 'gif')
VIDEO_PATH     = os.path.join(BASE_DIR, 'big-trouble-little-china.mp4')
SEG_MODEL_PATH = os.path.join(BASE_DIR, 'deeplabv3_257_mv2_1.0_257.tflite')
TW_LOG_PATH    = os.path.join(GIF_DIR, '.twitter_post_log.json')

# ─── Config Reader ─────────────────────────────────────────────────────────────
def read_config():
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#",";"))
    cfg.read(CONFIG_FILE)
    return (
        cfg.getfloat('General','start_time',    fallback=0.0),
        cfg.getint  ('General','display_time',  fallback=30),
        cfg.getint  ('General','frame_interval',fallback=30),
        cfg.getint  ('General','jitter',        fallback=5),
        cfg.getint  ('General','fat_bits',      fallback=2),
        cfg.get    ('General','update_mode',    fallback='DU').upper(),
        cfg.getint  ('General','cluster_size',  fallback=512),
        cfg.getfloat('General','chunk_pause',   fallback=0.05),
        cfg.getfloat('General','brightness',    fallback=1.8),
        cfg.getfloat('General','contrast',      fallback=1.5),
    )

# ─── Utilities ────────────────────────────────────────────────────────────────
def ensure_active_dir():
    if os.path.isdir(ACTIVE_DIR):
        for f in os.listdir(ACTIVE_DIR):
            try: os.remove(os.path.join(ACTIVE_DIR, f))
            except: pass
    else:
        os.makedirs(ACTIVE_DIR, exist_ok=True)

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
    _, h_in, w_in, _ = inp['shape']
    img = cv2.resize(im_color, (w_in, h_in))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if inp['dtype'] == np.uint8:
        scale, zp = inp['quantization']
        tensor = ((img/scale) + zp).astype(np.uint8)
    else:
        tensor = (img.astype(np.float32) / 255.0)

    interp.set_tensor(inp['index'], np.expand_dims(tensor,0))
    interp.invoke()
    raw = interp.get_tensor(out['index'])[0]

    C = raw.shape[-1]
    if C == 2:
        mask = (raw[:,:,1] > raw[:,:,0]).astype(np.uint8)
    else:
        cls = np.argmax(raw, axis=-1)
        mask = (cls == 15).astype(np.uint8)

    small = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(ACTIVE_DIR,'small-mask.png'), small*255)
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

# ─── GIF helpers — transparency-proof ─────────────────────────────────────────
PALETTE_LIST = [0,0,0, 255,255,255, 255,0,255] + [0,0,0]*253  # 0:black, 1:white, 2:dummy

def _apply_fixed_palette(frames_L):
    pal = Image.new("P", (1, 1))
    pal.putpalette(PALETTE_LIST)
    frames_P = []
    for imL in frames_L:
        p = imL.convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE)
        p.putpalette(PALETTE_LIST)
        p.info.pop("transparency", None)
        frames_P.append(p)
    return frames_P, pal

def _force_full_frame(frames_P, pal):
    w, h = frames_P[0].size
    forced = []
    for f in frames_P:
        canvas = Image.new("P", (w, h), 1)  # WHITE bg (idx 1)
        canvas.putpalette(PALETTE_LIST)
        canvas.paste(f, (0, 0))
        canvas.info.pop("transparency", None)
        canvas.info["background"] = 1
        canvas.info["disposal"] = 2
        forced.append(canvas)
    return forced

def _assert_no_dummy_index(frames_P):
    for i, f in enumerate(frames_P):
        if (np.array(f) == 2).any():
            print(f"⚠️ Frame {i} contains dummy transparent index 2! (should not happen)")

def _save_gif(frames_P, path, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dur = int(round(1000.0 / max(1, fps)))
    for f in frames_P:
        f.info.pop("transparency", None)
    _assert_no_dummy_index(frames_P)
    try:
        frames_P[0].save(
            path,
            save_all=True,
            append_images=frames_P[1:],
            duration=dur,
            loop=0,
            optimize=False,
            disposal=2,
            subrectangles=False,
            background=1,
            transparency=2,  # dummy index, never used
        )
    except TypeError:
        frames_P[0].save(
            path,
            save_all=True,
            append_images=frames_P[1:],
            duration=dur,
            loop=0,
            optimize=False,
            disposal=2,
            background=1,
            transparency=2,
        )
    print(f"🖼️ GIF written: {path}  (frames={len(frames_P)})")
    return path

def _save_gif_with_rotation(frames_P, out_path, fps, keep):
    if keep <= 1:
        return _save_gif(frames_P, out_path, fps)
    base = os.path.splitext(os.path.basename(out_path))[0]
    ts   = time.strftime("%Y%m%d-%H%M%S")
    ts_path = os.path.join(os.path.dirname(out_path), f"{base}-{ts}.gif")
    _save_gif(frames_P, ts_path, fps)
    try: shutil.copy2(ts_path, out_path)
    except Exception: pass
    try:
        folder = os.path.dirname(out_path)
        all_ts = [f for f in os.listdir(folder) if f.startswith(base + "-") and f.endswith(".gif")]
        all_ts_paths = [os.path.join(folder, f) for f in all_ts]
        all_ts_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for p in all_ts_paths[keep:]:
            try: os.remove(p)
            except: pass
    except Exception:
        pass
    print(f"🖼️ GIF written: {ts_path}  (kept newest {keep}, latest copied to {out_path})")
    return ts_path

# ─── Better validator (no more false alarms) ───────────────────────────────────
def _validate_edges(frames_P, content_box, strict=False):
    """
    If strict=False (default): Only warn when painted pixels appear OUTSIDE
    the known content box. White-at-the-edge is fine.
    If strict=True: old behavior (track rightmost non-white inside the box).
    """
    x0, y0, w, h = content_box
    x1 = x0 + w - 1
    y1 = y0 + h - 1

    lefts, rights = [], []
    drift_outside = False

    for f in frames_P:
        arr = np.array(f.convert('L'))

        # Anything painted outside the horizontal content range?
        left_slice  = arr[y0:y1+1, :x0]
        right_slice = arr[y0:y1+1, x1+1:]
        if (left_slice < 250).any() or (right_slice < 250).any():
            drift_outside = True

        if strict:
            sub = arr[y0:y1+1, x0:x1+1]
            xs_any = np.where(sub < 250)
            if xs_any[1].size:
                lefts.append(int(x0 + xs_any[1].min()))
                rights.append(int(x0 + xs_any[1].max()))
            else:
                lefts.append(None); rights.append(None)

    report = {
        "mode": "strict" if strict else "bounds-only",
        "content_box": {"x": x0, "y": y0, "w": w, "h": h},
    }

    if strict:
        # keep the old numbers for debugging
        valid_lefts  = [l for l in lefts  if l is not None]
        valid_rights = [r for r in rights if r is not None]
        if valid_lefts and valid_rights:
            report.update({
                "left_min": min(valid_lefts),
                "left_max": max(valid_lefts),
                "right_min": min(valid_rights),
                "right_max": max(valid_rights),
            })
        report["per_frame"] = [{"l": l, "r": r} for l, r in zip(lefts, rights)]

    # Print outcome
    if drift_outside:
        print("⚠️ EDGE DRIFT DETECTED: pixels appeared outside content box.", report)
    else:
        msg = "✅ Edges stable (geometry fixed"
        if not strict:
            msg += "; content near edges may be white"
        msg += ")."
        print(msg)

# ─── GIF builder ───────────────────────────────────────────────────────────────
def make_gif_from_chunks(
    chunks,
    disp_w, disp_h, x_off, y_off, nw, nh, fat_bits,
    out_path="gif/out.gif", fps=15, every_n=3, scale=1.0,
    settle_frames=2, tail_hold=5, keep=1,
    crop_letterbox=False, debug_guides=False,
    strict_edge_check=False,
):
    if abs(scale - 1.0) > 1e-9:
        print("ℹ️ gif_scale is ignored; rendering at exact panel size.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Base panel (L) + bars
    baseL = Image.new("L", (int(disp_w), int(disp_h)), 255)
    if y_off > 0:
        baseL.paste(0, (0, 0, disp_w, y_off))
    if (y_off + nh) < disp_h:
        baseL.paste(0, (0, y_off + nh, disp_w, disp_h))

    # Content canvas (L)
    contentL = Image.new("L", (int(nw), int(nh)), 255)

    # Real dither frame written in prepare_frame
    dither_path = os.path.join(ACTIVE_DIR, 'dithered.png')
    dfull = Image.open(dither_path).convert('L')

    frames_L = []
    first = baseL.copy()
    first.paste(contentL, (int(x_off), int(y_off)))
    frames_L.append(first.copy())

    painted = 0
    for cluster in chunks:
        for (px, py) in cluster:
            lx = int(px - x_off); ly = int(py - y_off)
            if lx < 0 or ly < 0: continue
            if (lx + fat_bits) > nw or (ly + fat_bits) > nh: continue
            tile = dfull.crop((px, py, px + fat_bits, py + fat_bits))
            contentL.paste(tile, (lx, ly))
        painted += 1
        if (painted % max(1, every_n)) == 0:
            frame = baseL.copy()
            frame.paste(contentL, (int(x_off), int(y_off)))
            frames_L.append(frame.copy())

    if (painted % max(1, every_n)) != 0:
        frame = baseL.copy()
        frame.paste(contentL, (int(x_off), int(y_off)))
        frames_L.append(frame.copy())

    for _ in range(max(0, settle_frames)): frames_L.append(frames_L[-1].copy())
    for _ in range(max(0, tail_hold)):     frames_L.append(frames_L[-1].copy())

    # Convert to P + enforce full frames
    frames_P, pal = _apply_fixed_palette(frames_L)
    frames_P = _force_full_frame(frames_P, pal)

    if debug_guides:
        gx0, gy0 = int(x_off), int(y_off)
        gx1, gy1 = gx0 + int(nw) - 1, gy0 + int(nh) - 1
        for i in range(len(frames_P)):
            f = frames_P[i]
            px = f.load()
            for x in range(gx0, gx1+1):
                px[x, gy0] = 0; px[x, gy1] = 0
            for y in range(gy0, gy1+1):
                px[gx0, y] = 0; px[gx1, y] = 0

    # New validator: no false alarm when edges are white
    _validate_edges(frames_P, (int(x_off), int(y_off), int(nw), int(nh)),
                    strict=strict_edge_check)

    # 1) Save UN-cropped
    uncropped_path = os.path.join(os.path.dirname(out_path), "uncropped_out.gif")
    _save_gif(frames_P, uncropped_path, fps)

    # 2) Crop 130 px from top & bottom
    h = frames_P[0].height; w = frames_P[0].width
    top_cut = 130; bot_cut = 130
    top_cut = max(0, min(top_cut, h // 2))
    bot_cut = max(0, min(bot_cut, h - top_cut))
    crop_box = (0, top_cut, w, h - bot_cut)

    cropped_P = []
    for f in frames_P:
        c = f.crop(crop_box)
        canvas = Image.new("P", c.size, 1)  # WHITE bg
        canvas.putpalette(PALETTE_LIST)
        canvas.paste(c, (0, 0))
        canvas.info.pop("transparency", None)
        canvas.info["background"] = 1
        canvas.info["disposal"] = 2
        cropped_P.append(canvas)

    return _save_gif_with_rotation(cropped_P, out_path, fps, keep)

# ─── Timestamp writer ─────────────────────────────────────────────────────────
def write_timestamp_file(gif_dir: str, movie_seconds: float):
    try:
        os.makedirs(gif_dir, exist_ok=True)
        wall_hms = time.strftime("%H:%M:%S", time.localtime())
        h = int(movie_seconds // 3600)
        m = int((movie_seconds % 3600) // 60)
        s = int(movie_seconds % 60)
        movie_hms = f"{h:02d}:{m:02d}:{s:02d}"
        with open(os.path.join(gif_dir, "timestamp.txt"), "w") as f:
            f.write(f"wallclock={wall_hms}\n")
            f.write(f"movie_time={movie_hms}\n")
            f.write(f"movie_seconds={movie_seconds:.2f}\n")
    except Exception as e:
        print(f"⚠️ timestamp.txt write failed: {e}")

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
    im_color_r = cv2.resize(frame_color,(nw,nh), interpolation=cv2.INTER_LANCZOS4)
    print(f"↔️ Resized to {nw}×{nh}")
    x_off,y_off = (disp_w-nw)//2,(disp_h-nh)//2

    interp, inp, out = seg_sess
    if interp:
        small_mask = get_silhouette_mask(seg_sess, im_color_r, (nw//fat_bits, nh//fat_bits), im_gray_r)
    else:
        small_mask = np.zeros((nh//fat_bits, nw//fat_bits),dtype=np.uint8)
    print("🔍 Silhouette mask computed")

    letterbox = Image.new('L',(disp_w,disp_h),255)
    if y_off > 0:
        letterbox.paste(0,(0,0,disp_w,y_off))
    if (y_off + nh) < disp_h:
        letterbox.paste(0,(0,y_off+nh,disp_w,disp_h))

    fb_off = letterbox.copy()
    small   = im_gray_r.resize((nw//fat_bits, nh//fat_bits), Image.Resampling.LANCZOS)
    dsmall  = atkinson_dither(small)
    dfull   = dsmall.resize((nw,nh), Image.Resampling.NEAREST)
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

    return letterbox, fb_off, chunks, sampled_t, x_off, y_off, nw, nh

# ─── Twitter posting + daily limiter ──────────────────────────────────────────
def _load_twitter_log():
    try:
        with open(TW_LOG_PATH, 'r') as f:
            return json.load(f)
    except:
        return {"date": "", "count": 0, "last_post_ts": 0}

def _save_twitter_log(data):
    os.makedirs(GIF_DIR, exist_ok=True)
    try:
        with open(TW_LOG_PATH, 'w') as f:
            json.dump(data, f)
    except:
        pass

def _can_post_today(daily_limit, min_interval_sec):
    data = _load_twitter_log()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if data.get("date") != today:
        data = {"date": today, "count": 0, "last_post_ts": 0}
        _save_twitter_log(data)
        return True, data
    if data.get("count", 0) >= max(0, daily_limit):
        return False, data
    if min_interval_sec > 0:
        elapsed = time.time() - data.get("last_post_ts", 0)
        if elapsed < min_interval_sec:
            return False, data
    return True, data

def _bump_post_counter(state):
    state["count"] = state.get("count", 0) + 1
    state["last_post_ts"] = time.time()
    _save_twitter_log(state)

def post_gif_to_twitter(twitter_cfg, gif_path, status_text=" "):
    try:
        import tweepy
    except ImportError:
        print("⚠️ Tweepy not installed. Skipping Twitter upload. (pip3 install --user tweepy)")
        return False

    api_key       = twitter_cfg.get("api_key", "")
    api_secret    = twitter_cfg.get("api_secret", "")
    access_token  = twitter_cfg.get("access_token", "")
    access_secret = twitter_cfg.get("access_secret", "")

    if not all([api_key, api_secret, access_token, access_secret]):
        print("⚠️ Missing Twitter credentials. Skipping upload.")
        return False

    try:
        auth1 = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
        api_v1 = tweepy.API(auth1)
        media = api_v1.media_upload(filename=gif_path)
        media_id = media.media_id

        client = tweepy.Client(
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_secret
        )
        resp = client.create_tweet(text=(status_text or " "), media_ids=[str(media_id)])
        if getattr(resp, "errors", None):
            print(f"⚠️ Twitter v2 error: {resp.errors}")
            return False
        print("🐦 Posted GIF to Twitter (v2).")
        return True

    except Exception as e:
        print(f"⚠️ Twitter post failed: {e}")
        return False

def twitter_notify_message(twitter_cfg, handle: str, text: str):
    try:
        import tweepy
    except ImportError:
        print("⚠️ Tweepy not installed. Skipping Twitter notify.")
        return False

    api_key       = twitter_cfg.get("api_key", "")
    api_secret    = twitter_cfg.get("api_secret", "")
    access_token  = twitter_cfg.get("access_token", "")
    access_secret = twitter_cfg.get("access_secret", "")

    if not all([api_key, api_secret, access_token, access_secret]):
        print("⚠️ Missing Twitter credentials. Skipping notify.")
        return False

    try:
        client = tweepy.Client(
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_secret
        )
        mention = f"@{handle} {text}".strip()
        resp = client.create_tweet(text=mention)
        if getattr(resp, "errors", None):
            print(f"⚠️ Twitter notify error: {resp.errors}")
            return False
        print("📣 Sent Twitter notification mention.")
        return True
    except Exception as e:
        print(f"⚠️ Twitter notify failed: {e}")
        return False

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    (start_t, display_time, frame_interval, jitter, fat_bits,
     update_mode, cluster_size, chunk_pause, brightness, contrast) = read_config()

    cfg = configparser.ConfigParser(inline_comment_prefixes=("#",";"))
    cfg.read(CONFIG_FILE)

    gifsec = cfg["GIF"] if "GIF" in cfg else {}
    gif_enabled       = gifsec.getboolean("make_gif", fallback=True)
    gif_fps           = gifsec.getint("gif_fps", fallback=4)
    gif_every_n       = gifsec.getint("gif_every_n", fallback=6)
    gif_scale         = gifsec.getfloat("gif_scale", fallback=1.0)  # ignored
    gif_settle        = gifsec.getint("gif_settle_frames", fallback=3)
    gif_tail_hold     = gifsec.getint("gif_tail_hold", fallback=10)
    gif_keep          = gifsec.getint("gif_keep", fallback=1)
    gif_crop_bars     = gifsec.getboolean("gif_crop_letterbox", fallback=False)
    gif_debug_guides  = str(gifsec.get("debug_guides","no")).strip().lower() in ("1","yes","true")
    gif_strict_edges  = str(gifsec.get("strict_edge_check","no")).strip().lower() in ("1","yes","true")
    gif_out_path      = os.path.join(GIF_DIR, "out.gif")

    twsec = cfg["Twitter"] if "Twitter" in cfg else {}
    twitter_enabled     = str(twsec.get("enable", "no")).strip().lower() in ("yes","true","1")
    twitter_daily_max   = int(twsec.get("daily_limit", 10))
    twitter_min_gap     = int(float(twsec.get("min_interval_sec", 0)))
    notify_on_restart   = str(twsec.get("notify_on_restart", "yes")).strip().lower() in ("yes","true","1")
    notify_handle       = twsec.get("notify_handle", "erminio").lstrip("@")
    restart_sleep_sec   = int(twsec.get("restart_sleep_sec", 7200))

    seg_sess = load_segmentation_model()
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/max(1.0, fps)

    epd = AutoEPDDisplay(vcom=-1.80)
    epd.clear()
    disp_w,disp_h = epd.width,epd.height

    print(f"▶️ Beta 1.0 start={start_t}s, br={brightness}, cr={contrast}, fat={fat_bits}, cluster={cluster_size}")
    t = start_t
    last_sample_t = -1.0

    while True:
        (lb, fb, chunks, t,
         x_off, y_off, nw, nh) = prepare_frame(
            cap,t,duration,
            disp_w,disp_h,
            fat_bits,frame_interval,jitter,cluster_size,
            brightness,contrast,seg_sess
        )

        wrapped = (last_sample_t >= 0) and (t < last_sample_t - 1.0)
        last_sample_t = t

        # ePaper draw
        epd.frame_buf = lb
        epd.draw_full(constants.DisplayModes.GC16)
        print("🟢 Letterbox drawn")

        for pts in chunks:
            for x0,y0 in pts:
                blk = fb.crop((x0,y0,x0+fat_bits,y0+fat_bits))
                epd.frame_buf.paste(blk,(x0,y0))
            epd.draw_partial(1)
            time.sleep(chunk_pause)

        print("✅ Frame completed")

        # GIF
        saved_path = None
        if gif_enabled:
            try:
                saved_path = make_gif_from_chunks(
                    chunks,
                    disp_w, disp_h, x_off, y_off, nw, nh, fat_bits,
                    out_path=gif_out_path,
                    fps=gif_fps, every_n=gif_every_n, scale=gif_scale,
                    settle_frames=gif_settle, tail_hold=gif_tail_hold,
                    keep=gif_keep, crop_letterbox=gif_crop_bars,
                    debug_guides=gif_debug_guides,
                    strict_edge_check=gif_strict_edges,
                )
                write_timestamp_file(GIF_DIR, t)
            except Exception as e:
                print(f"⚠️ GIF error: {e}")

        if twitter_enabled and (saved_path or os.path.isfile(gif_out_path)):
            ok, state = _can_post_today(twitter_daily_max, twitter_min_gap)
            if not ok:
                print("⏸️ Twitter: daily limit reached or interval not elapsed — skipping this one.")
            else:
                status = " "
                if post_gif_to_twitter(twsec, saved_path or gif_out_path, status):
                    _bump_post_counter(state)

        if wrapped and notify_on_restart:
            note = "THE END"
            print("🔔 Movie wrap detected — sending notification…")
            twitter_notify_message(twsec, notify_handle, note)
            print(f"⏸️ Sleeping for {restart_sleep_sec}s before restart…")
            time.sleep(restart_sleep_sec)
            t = 0.0
            last_sample_t = -1.0
            continue

        print(f"⏱️ Holding for {display_time}s")
        time.sleep(display_time)

if __name__ == "__main__":
    main()