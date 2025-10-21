import io, json, base64
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
import numpy as np
import torch
import torch.nn.functional as F
import bcrypt
import gdown  # ✅ safer model download

# =========================================================
# 📦 CONFIGURATION
# =========================================================
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model_ts.pt"
CFG_PATH = APP_DIR / "model_config.json"
USERS_FILE = APP_DIR / "users.txt"

# =========================================================
# ⚙️ LOAD MODEL CONFIG
# =========================================================
cfg = json.loads(CFG_PATH.read_text())
IMG_SIZE = int(cfg["img_size"])
MEAN = np.array(cfg["mean"], dtype=np.float32)
STD = np.array(cfg["std"], dtype=np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# 📥 DOWNLOAD MODEL FROM GOOGLE DRIVE
# =========================================================
MODEL_URL = "https://drive.google.com/uc?id=10ITq1tMY6k6-kwRa4_SHrNs8hVxaDl06"

try:
    if not MODEL_PATH.exists():
        print("⬇️ Downloading model from Google Drive using gdown...")
        gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)

    print("✅ Loading model...")
    model = torch.jit.load(str(MODEL_PATH), map_location=device).eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    model = None
    print("❌ Failed to load model:", e)

# =========================================================
# 🚀 FASTAPI SETUP
# =========================================================
app = FastAPI(title="VitGreen API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 Change to your Netlify/Frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 🔐 USER LOGIN FUNCTIONS
# =========================================================
def read_users():
    users = {}
    try:
        with open(USERS_FILE, "r") as f:
            for line in f:
                if ":" in line:
                    username, hashed_pw = line.strip().split(":", 1)
                    users[username] = hashed_pw
    except FileNotFoundError:
        print("❌ users.txt not found.")
    return users


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    users = read_users()
    print(f"🔍 Attempting login for: {username}")

    if username not in users:
        print("❌ Username not found.")
        raise HTTPException(status_code=401, detail="Invalid username or password")

    stored_hash = users[username].encode("utf-8")

    if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        print(f"✅ Login successful for {username}")
        return {"success": True, "message": f"Welcome, {username}!"}
    else:
        print("❌ Incorrect password.")
        raise HTTPException(status_code=401, detail="Invalid username or password")

# =========================================================
# 🌿 GVI ANALYSIS ENDPOINT
# =========================================================
def preprocess(pil_img: Image.Image):
    base = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(base).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    chw = np.transpose(arr, (2, 0, 1))
    x = torch.from_numpy(chw).unsqueeze(0).to(device)
    return x, base


def overlay_rgba(base: Image.Image, mask01: np.ndarray, alpha: float = 0.45) -> Image.Image:
    green = Image.new("RGBA", base.size, (0, 255, 0, int(alpha * 255)))
    m = Image.fromarray((mask01 * 255).astype(np.uint8), mode="L").resize(base.size, Image.NEAREST)
    green.putalpha(m)
    return Image.alpha_composite(base.convert("RGBA"), green).convert("RGB")


@app.post("/gvi")
async def gvi_endpoint(file: UploadFile = File(...)):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded on server")

        print("📸 Received image for analysis:", file.filename)

        content = await file.read()
        pil = Image.open(io.BytesIO(content)).convert("RGB")

        x, base = preprocess(pil)

        print("🧠 Running model inference...")
        with torch.no_grad():
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            if out.shape[-1] != IMG_SIZE or out.shape[-2] != IMG_SIZE:
                out = F.interpolate(out, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
            pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)

        print("🌿 Generating overlay...")
        gvi = float((pred == 1).sum() / pred.size)
        over = overlay_rgba(base, pred, alpha=0.45)

        buf = io.BytesIO()
        over.save(buf, format="JPEG", quality=90)
        overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        print(f"✅ GVI computed successfully: {gvi:.4f}")
        return JSONResponse({"gvi": gvi, "overlay": overlay_b64})

    except Exception as e:
        print("❌ GVI Error:", str(e))
        raise HTTPException(status_code=500, detail=f"GVI processing failed: {str(e)}")

# =========================================================
# 🏠 ROOT ROUTE
# =========================================================
@app.get("/")
def root():
    return {"message": "VitGreen API running successfully ✅"}