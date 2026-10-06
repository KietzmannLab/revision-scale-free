import os
import re
import glob
import json
import inspect

import numpy as np
import pandas as pd

from laion_fmri.config import dataset_initialize
from laion_fmri.discovery import get_rois, get_subjects
from laion_fmri.subject import load_subject
from laion_fmri.splits import get_split_masks

DATA_DIR = "/share/klab/datasets/optimized_datasets/laion_fmri_data"
OUT_DIR = "initial_exploration_out"
DETAIL_SUBJECT = "sub-01"
STREAM_ROIS = ["laiongeneral", "laionEVC", "laionventral", "laionlateral", "laiondorsal"]
NC_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.5]

os.makedirs(OUT_DIR, exist_ok=True)
dataset_initialize(DATA_DIR)


def section(title):
    print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72, flush=True)


def image_type(label):
    m = re.match(r"^(.*?)_LAION", str(label))
    return m.group(1) if m else str(label).split("_")[0]


def try_call(desc, fn):
    try:
        return fn()
    except Exception as e:
        print(f"  [could not get {desc}: {type(e).__name__}: {e}]")
        return None


subjects = get_subjects()
sub = load_subject(DETAIL_SUBJECT)

section("API probe")
print("Public attributes of Subject:", [a for a in dir(sub) if not a.startswith("_")])
print("get_betas signature:", try_call("signature", lambda: inspect.signature(sub.get_betas)))
print("get_split_masks signature:", try_call("signature", lambda: inspect.signature(get_split_masks)))

section("Q1  Subjects, sessions, trials, images")
rows = []
trials_all = {}
for s in subjects:
    sj = load_subject(s)
    sessions = sj.get_sessions()
    tinfo = sj.get_trial_info(session=sessions)
    t = pd.concat(list(tinfo.values()), ignore_index=True)
    t["img_type"] = t["label"].map(image_type)
    trials_all[s] = t
    rows.append({
        "subject": s,
        "n_sessions": len(sessions),
        "sessions": f"{sessions[0]}..{sessions[-1]}",
        "n_trials": len(t),
        "trials_per_session_min": t.groupby("session").size().min(),
        "trials_per_session_max": t.groupby("session").size().max(),
        "n_unique_images": t["label"].nunique(),
    })
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv(f"{OUT_DIR}/q1_subjects_sessions.csv", index=False)

t = trials_all[DETAIL_SUBJECT]
print(f"\nTrial table columns ({DETAIL_SUBJECT}):", list(t.columns))
print("\nImage types (trials / distinct images):")
print(t.groupby("img_type")["label"].agg(trials="size", images="nunique").to_string())
reps = t.groupby("label").size()
print("\nRepetitions per image (distribution):")
print(reps.value_counts().sort_index().to_string())
shared = t[t.img_type.str.startswith("shared")]
print("\nShared images: sessions each image appears in (distribution):")
print(shared.groupby("label")["session"].nunique().value_counts().sort_index().to_string())
per_ses = t.groupby(["label", "session"]).size()
print("\nRepetitions of the same image within one session (distribution over image-session pairs):")
print(per_ses.value_counts().sort_index().to_string())
print("\nWithin-session repeats by image type (max reps of one image in one session):")
print(per_ses.groupby("label").max().groupby(lambda l: image_type(l)).max().to_string())
ex = shared["label"].iloc[0]
print(f"\nExample shared image {ex}:")
print(t.loc[t.label == ex, ["session", "run", "beta_index"]].to_string(index=False))

shared_sets = {s: set(df.loc[df.img_type.str.startswith("shared"), "label"]) for s, df in trials_all.items()}
common = set.intersection(*shared_sets.values())
print(f"\nShared images common to all {len(subjects)} subjects: {len(common)}")
for s, st in shared_sets.items():
    print(f"  {s}: {len(st)} shared images")

section("Q2  ROIs / masks and their sizes")
for cat in [None, "laion", "retinotopy", "face", "place", "body", "object", "character", "motion"]:
    print(f"  category={cat}: {try_call('rois', lambda: get_rois(DETAIL_SUBJECT, category=cat))}")

first_ses = sub.get_sessions()[0]
full = sub.get_betas(session=first_ses)
print(f"\nFull betas {DETAIL_SUBJECT} {first_ses}: shape={full.shape}, dtype={full.dtype}")
roi_rows = []
for roi in STREAM_ROIS:
    b = try_call(roi, lambda: sub.get_betas(session=first_ses, roi=roi))
    if b is not None:
        roi_rows.append({"roi": roi, "n_voxels": b.shape[1]})
roi_df = pd.DataFrame(roi_rows)
print(roi_df.to_string(index=False))
roi_df.to_csv(f"{OUT_DIR}/q2_roi_sizes.csv", index=False)

section("Q3  Noise ceiling")
n_all = full.shape[1]
print(f"No threshold: {n_all} voxels")
for thr in NC_THRESHOLDS:
    b = try_call(f"nc={thr}", lambda: sub.get_betas(session=first_ses, nc_threshold=thr))
    if b is not None:
        print(f"  nc_threshold={thr}: {b.shape[1]} voxels ({100 * b.shape[1] / n_all:.1f}%)")
    b = try_call(f"laiongeneral nc={thr}",
                 lambda: sub.get_betas(session=first_ses, roi="laiongeneral", nc_threshold=thr))
    if b is not None:
        print(f"      within laiongeneral: {b.shape[1]} voxels")

nc_files = [f for f in glob.glob(f"{DATA_DIR}/derivatives/**/*{DETAIL_SUBJECT}*", recursive=True)
            if re.search(r"nc|noise|ceiling|reliab", os.path.basename(f), re.I)]
print("\nNoise-ceiling-like files on disk:", nc_files[:20] or "none found")

print("\nSplit-half reliability of shared images in laiongeneral (odd vs even repeats)...")
shared_b, shared_l = [], []
for ses in sub.get_sessions():
    b = np.asarray(sub.get_betas(session=ses, roi="laiongeneral"), dtype=np.float32)
    ti = sub.get_trial_info(session=ses)
    z = (b - b.mean(0)) / (b.std(0) + 1e-8)
    m = ti["label"].map(image_type).str.startswith("shared").values
    shared_b.append(z[m])
    shared_l.append(ti["label"].values[m])
B = np.concatenate(shared_b)
L = np.concatenate(shared_l)
imgs = [im for im in np.unique(L) if (L == im).sum() >= 2]
odd = np.stack([B[L == im][0::2].mean(0) for im in imgs])
even = np.stack([B[L == im][1::2].mean(0) for im in imgs])
oz = (odd - odd.mean(0)) / (odd.std(0) + 1e-8)
ez = (even - even.mean(0)) / (even.std(0) + 1e-8)
r = (oz * ez).mean(0)
print(f"  images used: {len(imgs)}, voxels: {B.shape[1]}")
print(f"  split-half r per voxel: median={np.median(r):.3f}, "
      f"pct r>0.1={100 * (r > 0.1).mean():.1f}%, pct r>0.3={100 * (r > 0.3).mean():.1f}%")
np.save(f"{OUT_DIR}/q3_splithalf_r_laiongeneral_{DETAIL_SUBJECT}.npy", r)

section("Q4  Space: native or template?")
deriv = f"{DATA_DIR}/derivatives/glmsingle-tedan/{DETAIL_SUBJECT}"
files = sorted(glob.glob(f"{deriv}/**/*", recursive=True))
exts = pd.Series([os.path.splitext(f)[1] if not f.endswith(".nii.gz") else ".nii.gz"
                  for f in files if os.path.isfile(f)]).value_counts()
print("File types in derivatives folder:\n", exts.to_string())
print("Example files:", [os.path.relpath(f, deriv) for f in files[:15]])
space_hits = sorted({m for f in files for m in re.findall(r"space-[A-Za-z0-9]+", f)})
print("BIDS 'space-' tags in filenames:", space_hits or "none")

for js in [f for f in files if f.endswith(".json")][:3]:
    print(f"\n{os.path.relpath(js, deriv)}:")
    try:
        print(json.dumps(json.load(open(js)), indent=1)[:1500])
    except Exception as e:
        print("  unreadable:", e)

try:
    import nibabel as nib
    for nf in [f for f in files if f.endswith((".nii", ".nii.gz"))][:3]:
        img = nib.load(nf)
        print(f"\n{os.path.relpath(nf, deriv)}: shape={img.shape}, "
              f"voxel size={img.header.get_zooms()[:3]}")
        print("affine:\n", np.round(img.affine, 2))
except ImportError:
    print("nibabel not installed")

section("Q5  Z-scored? Averaged?")
fb = np.asarray(full, dtype=np.float32)
vm, vs = fb.mean(0), fb.std(0)
print(f"Per-voxel mean over trials:  median={np.median(vm):.3f}, IQR=[{np.percentile(vm, 25):.3f}, {np.percentile(vm, 75):.3f}]")
print(f"Per-voxel std over trials:   median={np.median(vs):.3f}, IQR=[{np.percentile(vs, 25):.3f}, {np.percentile(vs, 75):.3f}]")
print(f"Overall value range: min={fb.min():.2f}, max={fb.max():.2f}, NaNs={np.isnan(fb).sum()}")
ti1 = sub.get_trial_info(session=first_ses)
print(f"Rows in betas = {fb.shape[0]}, rows in trial table = {len(ti1)}, "
      f"distinct images in session = {ti1['label'].nunique()}")
del fb, full

section("Q6  Train/test splits")
t = trials_all[DETAIL_SUBJECT]
split_names = ["random_0", "tau"] + [f"cluster_k5_{k}" for k in range(5)] + ["ood"]
split_rows = []
for name in split_names:
    kw = {"ood_types": ["shape", "unusual", "cropped"]} if name == "ood" else {}
    res = try_call(name, lambda: get_split_masks(t, name, pool="shared", **kw))
    if res is None:
        continue
    tr, te = np.asarray(res[0]), np.asarray(res[1])
    tr_imgs, te_imgs = set(t.loc[tr, "label"]), set(t.loc[te, "label"])
    split_rows.append({
        "split": name,
        "train_trials": int(tr.sum()), "test_trials": int(te.sum()),
        "unused_trials": int((~tr & ~te).sum()),
        "train_images": len(tr_imgs), "test_images": len(te_imgs),
        "image_overlap": len(tr_imgs & te_imgs),
        "test_img_types": dict(t.loc[te, "img_type"].value_counts()),
    })
split_df = pd.DataFrame(split_rows)
print(split_df.to_string(index=False))
split_df.to_csv(f"{OUT_DIR}/q6_splits_{DETAIL_SUBJECT}.csv", index=False)

print("\nInitial exploration finished.")
