import sys
from pathlib import Path
import streamlit as st
import numpy as np
import itertools, pickle, tempfile, shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.phash import compute_phash, hamming_distance
from embedding.resnet_embedder import ResNetEmbedder
from embedding.clip_embedder import CLIPEmbedder
from evaluation.gating import gate

@st.cache_resource
def load_models():
    resnet  = ResNetEmbedder()
    clipper = CLIPEmbedder()
    with open(PROJECT_ROOT / "data/processed/decider.pkl", "rb") as f:
        bundle = pickle.load(f)
    return resnet, clipper, bundle["clf"], bundle["T_hash"], bundle["T_high"], bundle["T_low"]

def cosine(a, b):
    return float(np.dot(a, b))

def is_duplicate(path_a, path_b, resnet, clipper, clf, T_hash):
    ha   = compute_phash(path_a)
    hb   = compute_phash(path_b)
    ph_d = hamming_distance(ha, hb)

    # Stage 1: pHash gate
    if ph_d > T_hash:
        return False, ph_d, None, None, "unique"

    # Stage 2: ResNet
    ea   = resnet.embed(path_a).numpy()
    eb   = resnet.embed(path_b).numpy()
    rs_s = cosine(ea, eb)

    decision = gate(ph_d, rs_s)

    if decision == "ACCEPT":
        return True, ph_d, rs_s, None, "resnet"

    if decision == "REJECT":
        return False, ph_d, rs_s, None, "unique"

    # Stage 3: CLIP
    ca   = clipper.embed(path_a).numpy()
    cb   = clipper.embed(path_b).numpy()
    cl_s = cosine(ca, cb)

    # Stage 4: LR
    label = int(clf.predict(np.array([[ph_d, rs_s, cl_s]]))[0])
    if label == 1:
        return True, ph_d, rs_s, cl_s, "clip_lr"
    else:
        return False, ph_d, rs_s, cl_s, "unique"

# ── page config ───────────────────────────────────────────────
st.set_page_config(page_title="EigenSoul — Duplicate Finder", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family:'Syne',sans-serif; background:#0a0a0f; color:#e8e8f0; }
.stApp { background:#0a0a0f; }
h1 { font-family:'Syne',sans-serif; font-weight:800; font-size:2.8rem !important;
     background:linear-gradient(135deg,#a78bfa,#60a5fa,#34d399);
     -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
h3 { font-family:'Syne',sans-serif; color:#a78bfa; }
.stat-box { background:#13131f; border:1px solid #2a2a3f; border-radius:12px; padding:1.2rem; text-align:center; }
.stat-num { font-size:2rem; font-weight:800; font-family:'Space Mono'; color:#a78bfa; }
.stat-lbl { font-size:0.72rem; color:#888; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }
.summary-row { background:#13131f; border:1px solid #2a2a3f; border-radius:12px;
               padding:1rem 1.5rem; margin:0.4rem 0; display:flex; justify-content:space-between; align-items:center; }
.summary-label { font-family:'Space Mono'; font-size:0.85rem; color:#ccc; }
.summary-val   { font-family:'Space Mono'; font-size:1.1rem; font-weight:700; }
.val-dup    { color:#a78bfa; }
.val-unique { color:#34d399; }
.val-total  { color:#60a5fa; }
.dup-card  { background:#1a0a2e; border:1px solid #a78bfa55; border-left:4px solid #a78bfa;
             border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.5rem; }
.pipeline-step { background:#0d0d1a; border:1px solid #2a2a3f; border-radius:8px;
                 padding:0.35rem 0.8rem; margin:0.2rem 0; font-family:'Space Mono'; font-size:0.75rem; }
.step-green  { border-left:3px solid #34d399; color:#34d399; }
.step-purple { border-left:3px solid #a78bfa; color:#a78bfa; }
.step-blue   { border-left:3px solid #60a5fa; color:#60a5fa; }
.badge { border-radius:6px; padding:2px 10px; font-size:0.72rem; font-family:'Space Mono'; }
.badge-phash  { background:#f59e0b22; color:#f59e0b; border:1px solid #f59e0b55; }
.badge-resnet { background:#60a5fa22; color:#60a5fa; border:1px solid #60a5fa55; }
.badge-clip   { background:#a78bfa22; color:#a78bfa; border:1px solid #a78bfa55; }
.mono { font-family:'Space Mono',monospace; font-size:0.78rem; color:#888; }
.stButton > button { background:linear-gradient(135deg,#a78bfa,#60a5fa); color:#0a0a0f;
    font-weight:700; font-family:'Syne',sans-serif; border:none; border-radius:10px;
    padding:0.6rem 2rem; font-size:1rem; }
hr { border-color:#2a2a3f; }
</style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1,8])
with col_logo:
    st.markdown("""
    <div style="display:flex; justify-content:center;">
        <img src="data:image/png;base64,{}"
             style="
                width:90px;
                height:90px;
                border-radius:50%;
                object-fit:cover;
             ">
    </div>
    """.format(
        __import__("base64")
        .b64encode(open("ui/logo.png", "rb").read())
        .decode()
    ), unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <div style="
        height:90px;
        display:flex;
        flex-direction:column;
        justify-content:center;
    ">
        <h1 style="margin:0;">EigenSoul</h1>
        <p style="margin:0;">
            <b>Near-Duplicate Image Detection</b>
            — pHash → ResNet → CLIP → Logistic Regression
        </p>
    </div>
    """, unsafe_allow_html=True)
st.divider()

resnet, clipper, clf, T_hash, T_high, T_low = load_models()

with st.sidebar:
    st.markdown("### ⚙️ Learned Thresholds")
    st.markdown(f'<p class="mono">T_hash &nbsp;: {T_hash:.1f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="mono">T_high &nbsp;: {T_high:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="mono">T_low &nbsp;&nbsp;: {T_low:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="mono">Ambiguous: {T_low:.4f} — {T_high:.4f}</p>', unsafe_allow_html=True)
    st.caption("Learned from Airbnb + Copydays training data")

st.markdown("### 📂 Upload Images")
st.markdown('<p class="mono">Supports JPG, PNG, WEBP — upload multiple images at once</p>', unsafe_allow_html=True)

uploaded = st.file_uploader("Drop images here", type=["jpg","jpeg","png","webp"],
                             accept_multiple_files=True, label_visibility="collapsed")
if uploaded:
    st.markdown(f'<p class="mono">✓ {len(uploaded)} images loaded</p>', unsafe_allow_html=True)

st.divider()

if uploaded and len(uploaded) >= 2:
    if st.button(f"🔍  Scan {len(uploaded)} Images for Duplicates"):

        tmpdir = Path(tempfile.mkdtemp())
        paths  = []
        for uf in uploaded:
            p = tmpdir / uf.name
            p.write_bytes(uf.read())
            paths.append(p)

        pairs = list(itertools.combinations(paths, 2))
        total = len(pairs)

        st.markdown(f"### 🔬 Scanning {total} image pairs…")
        bar    = st.progress(0)
        status = st.empty()

        # buckets
        dup_resnet  = []   # duplicate flagged by ResNet gate
        dup_clip_lr = []   # duplicate flagged by CLIP + LR
        unique_pairs = []  # all non-duplicates

        for i, (a, b) in enumerate(pairs):
            bar.progress((i + 1) / total)
            status.markdown(f'<p class="mono">Comparing {a.name} ↔ {b.name}</p>', unsafe_allow_html=True)
            try:
                dup, ph_d, rs_s, cl_s, decided_by = is_duplicate(a, b, resnet, clipper, clf, T_hash)
                entry = dict(a=a, b=b, ph_d=ph_d, rs_s=rs_s, cl_s=cl_s)
                if not dup:
                    unique_pairs.append(entry)
                elif decided_by == "resnet":
                    dup_resnet.append(entry)
                elif decided_by == "clip_lr":
                    dup_clip_lr.append(entry)
            except Exception as e:
                st.warning(f"Skipped {a.name} ↔ {b.name}: {e}")

        bar.empty(); status.empty()

        total_dups    = len(dup_resnet) + len(dup_clip_lr)
        total_unique  = len(unique_pairs)

        st.divider()

        # ── top stat boxes ─────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="stat-box"><div class="stat-num" style="color:#60a5fa">{total}</div><div class="stat-lbl">Total Pairs Scanned</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="stat-num" style="color:#a78bfa">{total_dups}</div><div class="stat-lbl">Total Duplicate Pairs</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-box"><div class="stat-num" style="color:#34d399">{total_unique}</div><div class="stat-lbl">Total Unique Pairs</div></div>', unsafe_allow_html=True)

        st.divider()

        # ── summary breakdown ──────────────────────────────────
        st.markdown("### 📊 Pipeline Breakdown")

        st.markdown(f"""
<div class="summary-row">
  <span class="summary-label">Total Pairs</span>
  <span class="summary-val val-total">{total}</span>
</div>
<div class="summary-row">
  <span class="summary-label">🔵 Duplicates flagged by ResNet</span>
  <span class="summary-val val-dup">{len(dup_resnet)}</span>
</div>
<div class="summary-row">
  <span class="summary-label">🟣 Duplicates flagged by CLIP + LR</span>
  <span class="summary-val val-dup">{len(dup_clip_lr)}</span>
</div>
<div class="summary-row">
  <span class="summary-label">✅ Total Unique Pairs</span>
  <span class="summary-val val-unique">{total_unique}</span>
</div>
<div class="summary-row" style="border:1px solid #a78bfa55;">
  <span class="summary-label" style="color:#a78bfa; font-weight:700;">🔴 Total Duplicate Pairs</span>
  <span class="summary-val val-dup" style="font-size:1.4rem;">{total_dups}</span>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # ── expandable image pair sections ─────────────────────
        def show_pairs(pairs_list, badge_class, badge_label, step_class, signals_fn):
            for d in pairs_list:
                col1, col2, col3 = st.columns([2, 2, 3])
                with col1:
                    st.image(str(d["a"]), caption=d["a"].name, width='stretch')
                with col2:
                    st.image(str(d["b"]), caption=d["b"].name, width='stretch')
                with col3:
                    st.markdown(signals_fn(d, badge_class, badge_label, step_class), unsafe_allow_html=True)

        def resnet_signals(d, badge_class, badge_label, step_class):
            return f"""
<div class="dup-card">
  <span class="badge {badge_class}">{badge_label}</span><br><br>
  <div class="pipeline-step step-green">pHash distance &nbsp;&nbsp; {d['ph_d']} &nbsp;(≤ {T_hash:.0f} ✓)</div>
  <div class="pipeline-step {step_class}">ResNet similarity &nbsp;{d['rs_s']:.4f} &nbsp;(≥ {T_high:.4f} ✓)</div>
  <div class="pipeline-step step-purple" style="color:#555; border-left-color:#555;">CLIP &nbsp;&nbsp; n/a (not needed)</div>
</div>"""

        def clip_lr_signals(d, badge_class, badge_label, step_class):
            cl = f"{d['cl_s']:.4f}" if d['cl_s'] is not None else "n/a"
            return f"""
<div class="dup-card">
  <span class="badge {badge_class}">{badge_label}</span><br><br>
  <div class="pipeline-step step-green">pHash distance &nbsp;&nbsp; {d['ph_d']} &nbsp;(≤ {T_hash:.0f} ✓)</div>
  <div class="pipeline-step step-blue">ResNet similarity &nbsp;{d['rs_s']:.4f} &nbsp;(ambiguous zone)</div>
  <div class="pipeline-step step-purple">CLIP + LR &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {cl} → DUPLICATE</div>
</div>"""

        if dup_resnet:
            with st.expander(f"🔵 Pairs flagged as Duplicate by ResNet — {len(dup_resnet)}"):
                show_pairs(dup_resnet, "badge-resnet", "DUPLICATE · ResNet", "step-blue", resnet_signals)

        if dup_clip_lr:
            with st.expander(f"🟣 Pairs flagged as Duplicate by CLIP + LR — {len(dup_clip_lr)}"):
                show_pairs(dup_clip_lr, "badge-clip", "DUPLICATE · CLIP+LR", "step-purple", clip_lr_signals)

        if total_dups == 0:
            st.success("✅ No duplicates found — all images are unique!")

        # ── All pairs view ─────────────────────────────────────
        all_pairs = [(d, "🔵 ResNet") for d in dup_resnet] + \
                    [(d, "🟣 CLIP+LR") for d in dup_clip_lr] + \
                    [(d, "✅ Unique") for d in unique_pairs]

        with st.expander(f"📋 View All {len(all_pairs)} Pairs"):
            for d, label in all_pairs:
                col1, col2, col3 = st.columns([2, 2, 3])
                with col1:
                    st.image(str(d["a"]), caption=d["a"].name, width='stretch')
                with col2:
                    st.image(str(d["b"]), caption=d["b"].name, width='stretch')
                with col3:
                    rs_str   = f"{d['rs_s']:.4f}" if d['rs_s']  is not None else "n/a"
                    cl_str   = f"{d['cl_s']:.4f}" if d['cl_s']  is not None else "n/a"
                    color    = "#a78bfa" if "Unique" not in label else "#34d399"
                    st.markdown(f"""
<div style="background:#13131f;border:1px solid #2a2a3f;border-left:4px solid {color};
            border-radius:12px;padding:1rem 1.2rem;">
  <span style="color:{color};font-family:'Space Mono';font-size:0.85rem;font-weight:700;">{label}</span><br><br>
  <div class="pipeline-step step-green">pHash distance &nbsp;&nbsp; {d['ph_d']}</div>
  <div class="pipeline-step step-blue">ResNet similarity &nbsp;{rs_str}</div>
  <div class="pipeline-step step-purple">CLIP similarity &nbsp;&nbsp; {cl_str}</div>
</div>""", unsafe_allow_html=True)


        shutil.rmtree(tmpdir, ignore_errors=True)

elif uploaded and len(uploaded) == 1:
    st.info("Upload at least 2 images to compare.")
else:
    st.markdown(f"""
<div style="border:2px dashed #a78bfa44;border-radius:16px;padding:2rem;text-align:center;
            background:#13131f;margin-bottom:1.5rem;">
  <p style="font-size:1.1rem;color:#a78bfa;font-weight:700;">Upload 2 or more images above</p>
  <p class="mono">Pipeline: pHash gate (T={T_hash:.0f}) → ResNet gate (T_high={T_high:.4f}, T_low={T_low:.4f}) → CLIP → LR</p>
</div>""", unsafe_allow_html=True)