"""Gemini Media2Text: 自分のAPIキーで画像・動画をテキスト化するComfyUIノード"""
import hashlib, io, json, os, tempfile, time
from datetime import datetime, timedelta, timezone
from fractions import Fraction

import av
import numpy as np
from PIL import Image

try:
    import folder_paths
    USER_DIR = folder_paths.get_user_directory()
except Exception:  # ComfyUI外でのテスト用
    USER_DIR = os.path.join(os.path.dirname(__file__), "_user")

BASE_DIR = os.path.join(USER_DIR, "gemini_media2text")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
FILES_INDEX = os.path.join(BASE_DIR, "files_index.json")
INLINE_LIMIT = 18 * 1024 * 1024
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite",
                   "gemini-3.1-flash-lite", "gemini-4-argon"]
BLOCK_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY", "OTHER"}


# ---------- 設定・キー ----------
def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_api_key():
    key = os.environ.get("GEMINI_API_KEY") or load_config().get("api_key")
    if not key:
        raise RuntimeError("APIキーがありません。環境変数 GEMINI_API_KEY か "
                           f"{CONFIG_PATH} の api_key に設定してください。")
    return key


def make_client():
    from google import genai
    return genai.Client(api_key=get_api_key())


def list_models():
    try:
        names = [m.name.removeprefix("models/") for m in make_client().models.list()
                 if "generateContent" in (m.supported_actions or [])
                 and m.name.removeprefix("models/").startswith("gemini")]
        return sorted(set(names)) or FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


# ---------- メディア処理 ----------
def to_uint8(frames):
    """IMAGEテンソル[N,H,W,C](0-1 float) → uint8 ndarray"""
    arr = frames.cpu().numpy() if hasattr(frames, "cpu") else np.asarray(frames)
    return (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)


def trim_range(n_frames, src_fps, unit, start, end):
    if unit == "frames":
        s, e = int(start), (int(end) if end > 0 else n_frames)
    else:
        s, e = int(round(start * src_fps)), (int(round(end * src_fps)) if end > 0 else n_frames)
    s, e = max(0, s), min(n_frames, e)
    if e <= s:
        raise ValueError(f"区間が空です（start={start}, end={end}, unit={unit}, 総フレーム={n_frames}）")
    return s, e


def resample_idx(n, src_fps, dst_fps):
    if dst_fps <= 0 or dst_fps >= src_fps:
        return list(range(n))
    dur = n / src_fps
    return sorted({min(n - 1, int(round(k / dst_fps * src_fps)))
                   for k in range(max(1, int(dur * dst_fps)))})


def target_size(w, h, mp, even=False):
    scale = min(1.0, (mp * 1_000_000 / (w * h)) ** 0.5) if mp > 0 else 1.0
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    if even:
        nw, nh = max(2, nw - nw % 2), max(2, nh - nh % 2)
    return nw, nh


def encode_jpeg(img_u8, mp, quality):
    im = Image.fromarray(img_u8[..., :3])
    im = im.resize(target_size(*im.size, mp), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def encode_h264(frames_u8, fps, mp, crf, audio=None):
    """frames_u8:[N,H,W,C]、audio:(waveform ndarray[C,T] float, sample_rate) or None"""
    h, w = frames_u8.shape[1:3]
    tw, th = target_size(w, h, mp, even=True)
    fr = Fraction(fps).limit_denominator(1000)
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as out:
        codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
        vs = out.add_stream(codec, rate=fr)
        vs.width, vs.height, vs.pix_fmt = tw, th, "yuv420p"
        if codec == "libx264":
            vs.options = {"crf": str(int(crf)), "preset": "medium"}
        else:  # crf相当がないためqscaleで近似
            vs.options = {"qscale": str(max(2, min(31, int(crf) // 2)))}
        astream = None
        if audio is not None:
            wav, sr = audio
            astream = out.add_stream("aac", rate=int(sr))
            astream.layout = "stereo" if wav.shape[0] >= 2 else "mono"
        for f in frames_u8:
            im = Image.fromarray(f[..., :3]).resize((tw, th), Image.LANCZOS)
            for p in vs.encode(av.VideoFrame.from_image(im)):
                out.mux(p)
        for p in vs.encode():
            out.mux(p)
        if astream is not None:
            wav, sr = audio
            wav = wav[:2].astype(np.float32)
            fsz = astream.codec_context.frame_size or 1024
            for i in range(0, wav.shape[1], fsz):
                chunk = np.ascontiguousarray(wav[:, i:i + fsz])
                af = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=astream.layout.name)
                af.sample_rate = int(sr)
                for p in astream.encode(af):
                    out.mux(p)
            for p in astream.encode():
                out.mux(p)
        codec_used = codec
    return buf.getvalue(), codec_used, (tw, th)


# ---------- キャッシュ ----------
def sha256(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def cache_read(key):
    p = os.path.join(CACHE_DIR, key, "text.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read()
    return None


def cache_write(key, text, request, response):
    d = os.path.join(CACHE_DIR, key)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "text.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(d, "request.json"), "w", encoding="utf-8") as f:
        json.dump(request, f, ensure_ascii=False, indent=1)
    with open(os.path.join(d, "response.json"), "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=1, default=str)


def files_index_load():
    try:
        with open(FILES_INDEX, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def upload_or_reuse(client, data, mime, upload_key):
    idx = files_index_load()
    ent = idx.get(upload_key)
    now = datetime.now(timezone.utc)
    if ent and datetime.fromisoformat(ent["expiration_time"]) - now > timedelta(hours=1):
        return ent["uri"], True
    suffix = ".mp4" if mime == "video/mp4" else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(data)
        path = tf.name
    try:
        f = client.files.upload(file=path, config={"mime_type": mime})
        while getattr(f.state, "name", str(f.state)) == "PROCESSING":
            time.sleep(2)
            f = client.files.get(name=f.name)
        if getattr(f.state, "name", str(f.state)) != "ACTIVE":
            raise RuntimeError(f"Files APIの処理に失敗: {f.state} {f.error}")
    finally:
        os.remove(path)
    exp = f.expiration_time or (now + timedelta(hours=47))
    idx[upload_key] = {"uri": f.uri, "name": f.name, "expiration_time": exp.isoformat()}
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(FILES_INDEX, "w", encoding="utf-8") as fp:
        json.dump(idx, fp, ensure_ascii=False, indent=1)
    return f.uri, False


def estimate_usd(model, usage):
    price = load_config().get("prices", {}).get(model)
    if not price or not usage:
        return None
    inp = (usage.get("prompt_token_count") or 0) * price["input"]
    out = ((usage.get("candidates_token_count") or 0) + (usage.get("thoughts_token_count") or 0)) * price["output"]
    return round((inp + out) / 1_000_000, 6)


# ---------- ノード ----------
class GeminiMedia2Text:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list_models(), {"default": "gemini-3.8-flash"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "fps": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "total_megapixels": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 16.0, "step": 0.01}),
                "h264_crf": ("INT", {"default": 23, "min": 0, "max": 51}),
                "jpeg_quality": ("INT", {"default": 90, "min": 1, "max": 100}),
                "media_resolution": (["unspecified", "low", "medium", "high"],),
                "thinking_level": (["MEDIUM", "MINIMAL", "LOW", "HIGH", "unspecified"],),
                "max_output_tokens": ("INT", {"default": 8192, "min": 64, "max": 65536}),
                "include_audio": ("BOOLEAN", {"default": False}),
                "range_unit": (["seconds", "frames"],),
                "range_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e7, "step": 0.01}),
                "range_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e7, "step": 0.01}),
                "source_fps": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 240.0, "step": 0.01}),
                "cache_mode": (["normal", "force_refresh", "cache_only"],),
                "on_block": (["passthrough", "raise"],),
            },
            "optional": {
                "video": ("VIDEO",),
                "images": ("IMAGE",),
                "fallback_text": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "success", "status", "usage", "cache_key")
    FUNCTION = "run"
    CATEGORY = "Gemini"

    def run(self, model, prompt, system_prompt, fps, total_megapixels, h264_crf, jpeg_quality,
            media_resolution, thinking_level, max_output_tokens, include_audio, range_unit,
            range_start, range_end, source_fps, cache_mode, on_block,
            video=None, images=None, fallback_text=""):
        fallback_text = fallback_text or ""

        def fail(status, usage=None, key=""):
            if on_block == "raise" and status != "cache_miss":
                raise RuntimeError(f"Gemini Media2Text: {status}")
            return (fallback_text, False, status, json.dumps(usage or {}, ensure_ascii=False), key)

        if not prompt.strip():
            raise ValueError("prompt が空です")

        # 1) 正規化
        audio = None
        if video is not None:
            comp = video.get_components()
            frames, src_fps = to_uint8(comp.images), float(comp.frame_rate)
            if include_audio and comp.audio is not None:
                wav = comp.audio["waveform"]
                wav = wav[0].cpu().numpy() if hasattr(wav, "cpu") else np.asarray(wav)[0]
                audio = (wav, int(comp.audio["sample_rate"]))
        elif images is not None:
            frames, src_fps = to_uint8(images), float(source_fps)
        else:
            raise ValueError("video か images のどちらかを接続してください")
        is_still = frames.shape[0] == 1

        # 2) 区間切り出し
        if not is_still:
            s, e = trim_range(frames.shape[0], src_fps, range_unit, range_start, range_end)
            frames = frames[s:e]
            if audio is not None:
                wav, sr = audio
                audio = (wav[:, int(s / src_fps * sr):int(e / src_fps * sr)], sr)
            range_info = {"start_sec": s / src_fps, "end_sec": e / src_fps,
                          "start_frame": s, "end_frame": e}
        else:
            range_info = None

        # 3) ハッシュ・キャッシュキー
        media_hash = sha256(frames.tobytes(), frames.shape,
                            audio[0].astype(np.float32).tobytes() if audio is not None else b"")
        enc_params = (fps, total_megapixels, h264_crf if not is_still else jpeg_quality, src_fps)
        key = sha256(media_hash, model, prompt, system_prompt, *enc_params, json.dumps(range_info),
                     media_resolution, thinking_level, max_output_tokens, include_audio)

        # 4) キャッシュ判定
        if cache_mode != "force_refresh":
            hit = cache_read(key)
            if hit is not None:
                return (hit, True, "cache_hit",
                        json.dumps({"cache_hit": True, "estimated_usd": 0}, ensure_ascii=False), key)
            if cache_mode == "cache_only":
                return fail("cache_miss", {"cache_hit": False}, key)

        # 5) 再エンコード
        if is_still:
            data, mime, codec, size, n_sent = encode_jpeg(frames[0], total_megapixels, jpeg_quality), \
                "image/jpeg", "jpeg", None, 1
        else:
            idx = resample_idx(frames.shape[0], src_fps, fps)
            send_fps = fps if fps < src_fps else src_fps
            data, codec, size = encode_h264(frames[idx], send_fps, total_megapixels, h264_crf, audio)
            mime, n_sent = "video/mp4", len(idx)

        # 6) 送信
        from google.genai import errors, types
        try:
            client = make_client()
            vmeta = None if is_still else types.VideoMetadata(fps=send_fps)
            used_files_api, reused = len(data) >= INLINE_LIMIT, False
            if used_files_api:
                uri, reused = upload_or_reuse(client, data, mime, sha256(media_hash, *enc_params,
                                                                         json.dumps(range_info), include_audio))
                part = types.Part(file_data=types.FileData(file_uri=uri, mime_type=mime), video_metadata=vmeta)
            else:
                part = types.Part(inline_data=types.Blob(data=data, mime_type=mime), video_metadata=vmeta)
            cfg = {"max_output_tokens": max_output_tokens}
            if system_prompt.strip():
                cfg["system_instruction"] = system_prompt
            if media_resolution != "unspecified":
                cfg["media_resolution"] = f"MEDIA_RESOLUTION_{media_resolution.upper()}"
            if thinking_level != "unspecified":
                cfg["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
            resp = client.models.generate_content(model=model, contents=[part, prompt],
                                                  config=types.GenerateContentConfig(**cfg))
        except errors.ClientError as ex:
            msg = str(ex)
            blocked = any(w in msg.upper() for w in ("SAFETY", "PROHIBITED", "BLOCK"))
            return fail("blocked" if blocked else f"error:{ex.code} {msg[:200]}", key=key)
        except Exception as ex:
            return fail(f"error:{type(ex).__name__} {str(ex)[:200]}", key=key)

        # 7) 応答処理
        usage = resp.usage_metadata.model_dump(exclude_none=True) if resp.usage_metadata else {}
        info = {**usage, "cache_hit": False, "estimated_usd": estimate_usd(model, usage),
                "frames_sent": n_sent, "send_fps": None if is_still else send_fps, "codec": codec,
                "size": size, "bytes": len(data), "files_api": used_files_api, "files_api_reused": reused,
                "range": range_info}
        pf = resp.prompt_feedback
        finish = resp.candidates[0].finish_reason.name if resp.candidates and resp.candidates[0].finish_reason else None
        text = resp.text or ""
        if (pf and pf.block_reason) or (finish in BLOCK_REASONS) or not text:
            info["block_reason"] = str(pf.block_reason) if pf and pf.block_reason else finish
            return fail("blocked" if (pf and pf.block_reason) or finish in BLOCK_REASONS else "error:empty_response",
                        info, key)

        # 8) 保存
        request_log = {"model": model, "prompt": prompt, "system_prompt": system_prompt, "fps": fps,
                       "total_megapixels": total_megapixels, "h264_crf": h264_crf, "jpeg_quality": jpeg_quality,
                       "media_resolution": media_resolution, "thinking_level": thinking_level,
                       "max_output_tokens": max_output_tokens, "include_audio": include_audio,
                       "range": range_info, "media_hash": media_hash, "source_fps": src_fps,
                       "time": datetime.now(timezone.utc).isoformat()}
        cache_write(key, text, request_log, resp.model_dump(mode="json", exclude_none=True))
        return (text, True, "ok", json.dumps(info, ensure_ascii=False, default=str), key)

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan") if kw.get("cache_mode") == "force_refresh" else ""


NODE_CLASS_MAPPINGS = {"GeminiMedia2Text": GeminiMedia2Text}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiMedia2Text": "Gemini Media2Text (API Key)"}
