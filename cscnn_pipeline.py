"""
CSCNN: decorrelation contrast stretching and a lightweight CNN for oral histopathology.

Runs the complete analysis for each dataset: image-enhancement study, augmentation-order
experiment, CSCNN with ablations, 16 ImageNet-pretrained baselines, figures and tables.

Usage: edit the SETTINGS block below, then run  python cscnn_pipeline.py
"""
import glob
import json
import os
import re
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

# =====================================================================
#                              SETTINGS  (edit here)
# =====================================================================
# ---------------- choose the dataset for this run ----------------
DATASET = "tongue"        # "tongue"  (712 tongue biopsy images, Mendeley Data)
                          # "rahman"  (public: normal vs OSCC, 100x/400x)
                          # "ndbufes" (public: OSCC / leukoplakia with / without dysplasia,
                          #            patient-level split using the dataset's patient IDs)
WORK_ROOT = "/content/drive/MyDrive/CSCNN_work"   # all outputs are written here, one sub-folder per dataset
DATASET_DIRS = {
    "tongue": "/content/drive/MyDrive/TONGUE-RAW",
    "rahman": "/content/drive/MyDrive/ftmp4cvtmb-1",
    "ndbufes": "/content/drive/MyDrive/NDB-UFES/NDB-UFES An oral cancer and leukoplakia "
               "dataset composed of histopathological images and patient data",
}
DATASET_DIR = DATASET_DIRS[DATASET]
WORK_DIR = f"/content/drive/MyDrive/CSCNN_work/{DATASET}"   # separate outputs per dataset
METADATA_CSV = f"{WORK_DIR}/metadata.csv"         # created automatically
RESULTS_DIR = f"{WORK_DIR}/results_v2"
FIGURES_DIR = f"{WORK_DIR}/figures_v2"
LOCAL_COPY = f"/content/data_{DATASET}"           # images copied here (much faster than Drive)

# Without patient IDs each image is its own group; exact duplicate files are detected
# automatically and kept in the same fold.
GROUP_EACH_IMAGE = True
PATIENT_ID_REGEX = None
SLIDE_ID_REGEX = None
MAGNIFICATION_REGEX = r"(4x|10x|40x|100x|400x)"   # read from file/folder names when present

DATASETS_TO_RUN = ["tongue", "ndbufes", "rahman"]   # processed one after another
STEPS = ["enhancement", "leakage_demo", "cscnn", "ablations", "baselines", "figures"]
LEAKAGE_DEMO_DATASETS = ["tongue"]   # augmentation-order experiment
QUICK_TEST = False        # True = 2 epochs, 1 seed, CSCNN + MobileNetV2 only (checks setup)

IMG_SIZE = 224
BATCH = 16                # batch size for all models
MIXED_PRECISION = True    # faster on A100 / V100 / T4; outputs kept in float32
CACHE_SIDE = 320          # images pre-resized once (longest side) for fast training
EPOCHS = 100
PATIENCE = 15             # early stopping on validation balanced accuracy
SEEDS = [1, 2, 3, 4, 5]   # used only when CV_FOLDS is None (repeated random splits)
CV_FOLDS = 5              # 5-fold CV for CSCNN and ablations: every image is tested once
CSCNN_FOLDS = None        # None = all 5 folds for CSCNN and ablations; 1 = fold 1 only (quick test)
BASELINE_FOLDS = 1        # the 16 pretrained baselines run on fold 1 only (to save time);
                          # model comparisons are then made on that same fold for all models

# Image-enhancement evaluation
ENH_MAX_SIDE = 1024       # images are resized so the longest side is at most this (speed)
ENH_BRISQUE = True        # no-reference quality score (installs the small 'piq' package)

# CSCNN (proposed) configuration
CSCNN_BLOCKS = 5
CSCNN_DROPOUT = 0.3
CSCNN_BATCHNORM = True

# Folder name -> class name used in figures and tables.
CLASS_NAMES_OVERRIDE = {"TONGUE INFLAMMATORY": "Chronic inflammatory",
                        "TONGUE MILD DYSPLASIA": "Mild epithelial dysplasia",
                        "TONGUE MODERATELY DIFFERENTIATED-SCC": "Moderately diff. SCC",
                        "TONGUE-NO EVIDENCE OF MALIGNANCY": "No evidence of malignancy",
                        "Normal": "Normal oral epithelium", "OSCC": "OSCC",
                        "Leukoplakia with dysplasia": "Leukoplakia with dysplasia",
                        "Leukoplakia without dysplasia": "Leukoplakia without dysplasia"}
CLASS_ABBR_OVERRIDE = {"TONGUE INFLAMMATORY": "INF", "TONGUE MILD DYSPLASIA": "MED",
                       "TONGUE MODERATELY DIFFERENTIATED-SCC": "MD-SCC",
                       "TONGUE-NO EVIDENCE OF MALIGNANCY": "NEM",
                       "Normal": "Normal", "OSCC": "OSCC",
                       "Leukoplakia with dysplasia": "L+D", "Leukoplakia without dysplasia": "L-D"}
# =====================================================================

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:            # lets the data utilities be imported without TF
    tf = None

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
CSCNN_FILTERS = (16, 32, 32, 64, 64)
PROPOSED = "cscnn_v3_5blk_do0.3"          # reset in make_all_figures()

# (tf.keras.applications sub-module, constructor)
BASELINES = {
    "vgg16": ("vgg16", "VGG16"),
    "vgg19": ("vgg19", "VGG19"),
    "resnet50": ("resnet", "ResNet50"),
    "inceptionv3": ("inception_v3", "InceptionV3"),
    "xception": ("xception", "Xception"),
    "mobilenetv2": ("mobilenet_v2", "MobileNetV2"),
    "resnet101": ("resnet", "ResNet101"),
    "resnet50v2": ("resnet_v2", "ResNet50V2"),
    "inceptionresnetv2": ("inception_resnet_v2", "InceptionResNetV2"),
    "densenet121": ("densenet", "DenseNet121"),
    "densenet169": ("densenet", "DenseNet169"),
    "mobilenetv3small": ("mobilenet_v3", "MobileNetV3Small"),
    "mobilenetv3large": ("mobilenet_v3", "MobileNetV3Large"),
    "efficientnetb0": ("efficientnet", "EfficientNetB0"),
    "efficientnetv2b0": ("efficientnet_v2", "EfficientNetV2B0"),
    "convnexttiny": ("convnext", "ConvNeXtTiny"),
}


# ================================================================ metadata
def _label_from_folder(folder):
    """Class label from the image's folder name (dataset-specific)."""
    name = os.path.basename(folder)
    if globals().get("DATASET") == "rahman":
        return "OSCC" if "oscc" in name.lower() else "Normal"
    return name


def make_metadata(dataset_dir, out_csv="metadata.csv"):
    """Walk dataset_dir/<class>/<image>, build metadata.csv, group exact duplicates."""
    import hashlib
    rows = []
    for root, _, files in os.walk(dataset_dir):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXT):
                p = os.path.join(root, f)
                with open(p, "rb") as fh:
                    md5 = hashlib.md5(fh.read()).hexdigest()
                mag = re.search(r"(\d+x)", os.path.relpath(p, dataset_dir), re.IGNORECASE)
                rows.append({"filepath": p, "label": _label_from_folder(root),
                             "patient_id": "", "slide_id": "",
                             "magnification": mag.group(1).lower() if mag else "", "md5": md5})
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit(f"[ERROR] no images found in {dataset_dir}")
    # exact duplicate files
    dup = df[df.duplicated("md5", keep=False)]
    if len(dup):
        conflict = dup.groupby("md5")["label"].nunique()
        bad = conflict[conflict > 1].index
        dup.sort_values("md5").to_csv(out_csv.replace(".csv", "_duplicates.csv"), index=False)
        print(f"[WARN] {len(dup)} files are exact duplicates of another file "
              f"(listed in {os.path.basename(out_csv).replace('.csv', '_duplicates.csv')}).")
        if len(bad):
            print(f"[WARN] {len(bad)} identical images appear under DIFFERENT classes -> removed.")
            df = df[~df["md5"].isin(bad)]
    if GROUP_EACH_IMAGE:
        df["patient_id"] = "img_" + df["md5"].str[:16]      # duplicates share one group
        df["slide_id"] = df["patient_id"]
    df.to_csv(out_csv, index=False)
    print(f"[INFO] {len(df)} images -> {out_csv}\n{df['label'].value_counts().to_string()}")


def load_metadata(csv):
    meta = pd.read_csv(csv, dtype={"patient_id": str, "slide_id": str})
    bad = meta["patient_id"].isna() | (meta["patient_id"].str.strip() == "")
    if bad.any():
        raise ValueError(f"{bad.sum()} rows have no patient_id; fill them first.")
    if meta["slide_id"].isna().all():
        meta["slide_id"] = meta["patient_id"]
    multi = meta.groupby("patient_id")["label"].nunique()
    if (multi > 1).any():
        print(f"[WARN] {int((multi > 1).sum())} patients have >1 label.")
    return meta


# ================================================================== splits
def check_no_leakage(train, val, test):
    for key in ("patient_id", "slide_id"):
        a, b, c = (set(d[key]) for d in (train, val, test))
        assert not (a & b or a & c or b & c), f"{key} overlap between subsets!"


def iter_splits(meta, seed, cv=None):
    """Yield (fold, train, val, test); every patient is confined to one subset."""
    outer = StratifiedGroupKFold(n_splits=cv or 5, shuffle=True, random_state=seed)
    folds = list(outer.split(meta, meta["label"], meta["patient_id"]))
    for k, (trval_idx, test_idx) in enumerate(folds if cv else folds[:1]):
        trval, test = meta.iloc[trval_idx], meta.iloc[test_idx]
        inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed)
        tr_idx, val_idx = next(inner.split(trval, trval["label"], trval["patient_id"]))
        train, val = trval.iloc[tr_idx], trval.iloc[val_idx]
        check_no_leakage(train, val, test)
        yield f"fold{k}", train, val, test


def split_summary(train, val, test):
    out = []
    for name, d in (("train", train), ("val", val), ("test", test)):
        g = d.groupby("label").agg(patients=("patient_id", "nunique"),
                                   images=("filepath", "count")).reset_index()
        g.insert(0, "subset", name)
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ================================================= decorrelation stretching
def decorrelation_stretch_np(img, target_sigma=50.0, eps=1e-3):
    """NumPy reference implementation (used for figures). img: HxWx3, 0-255."""
    h, w, c = img.shape
    x = img.reshape(-1, c).astype(np.float64)
    mean = x.mean(0)
    xc = x - mean
    cov = xc.T @ xc / (len(x) - 1) + eps * np.eye(c)
    evals, evecs = np.linalg.eigh(cov)
    t = evecs @ np.diag(target_sigma / np.sqrt(np.maximum(evals, eps))) @ evecs.T
    return np.clip(xc @ t + mean, 0, 255).reshape(h, w, c).astype(np.float32)


if tf is not None:
    @tf.keras.utils.register_keras_serializable(package="cscnn")
    class DecorrelationStretch(layers.Layer):
        """Per-image decorrelation contrast stretch (Gillespie et al., 1986).
        For each image: PCA of the RGB channel covariance, rescale every principal
        component to a common standard deviation `target_sigma`, rotate back.
        Removes inter-channel (haematoxylin/eosin) correlation and equalises
        contrast. No trainable weights; input and output are in [0, 255]."""

        def __init__(self, target_sigma=50.0, eps=1e-3, **kw):
            super().__init__(**kw)
            self.target_sigma, self.eps = float(target_sigma), float(eps)

        def call(self, x):
            x = tf.cast(x, tf.float32)
            shape = tf.shape(x)
            flat = tf.reshape(x, [shape[0], -1, 3])                      # B,N,3
            mean = tf.reduce_mean(flat, axis=1, keepdims=True)
            xc = flat - mean
            n = tf.cast(tf.shape(flat)[1], tf.float32)
            cov = tf.matmul(xc, xc, transpose_a=True) / (n - 1.0)        # B,3,3
            cov += self.eps * tf.eye(3)                                    # broadcasts over batch
            evals, evecs = tf.linalg.eigh(cov)
            scale = self.target_sigma / tf.sqrt(tf.maximum(evals, self.eps))
            t = tf.matmul(evecs * scale[:, None, :], evecs, transpose_b=True)
            y = tf.clip_by_value(tf.matmul(xc, t) + mean, 0.0, 255.0)
            return tf.stop_gradient(tf.reshape(y, shape))              # fixed transform

        def get_config(self):
            cfg = super().get_config()
            cfg.update(target_sigma=self.target_sigma, eps=self.eps)
            return cfg

    class BackbonePreprocess(layers.Layer):
        """Applies a keras.applications preprocess_input to 0-255 inputs."""

        def __init__(self, module_name, **kw):
            super().__init__(**kw)
            self.module_name = module_name

        def call(self, x):
            mod = getattr(tf.keras.applications, self.module_name)
            return mod.preprocess_input(tf.cast(x, tf.float32))

        def get_config(self):
            cfg = super().get_config()
            cfg.update(module_name=self.module_name)
            return cfg


# ================================================================== models
def build_cscnn(input_shape, classes, blocks=5, stretch=True, batchnorm=True,
                dropout=0.3, target_sigma=50.0):
    """Lightweight CSCNN: [DecorrelationStretch] -> `blocks` x (Conv3x3 -> BN -> ReLU
    -> MaxPool2x2) with filters 16-32-32-64-64 -> GlobalAvgPool -> Dropout -> softmax."""
    inp = layers.Input(shape=input_shape, name="image_0_255")
    x = DecorrelationStretch(target_sigma, name="decorrelation_stretch")(inp) if stretch else inp
    x = layers.Rescaling(1.0 / 255, name="rescale")(x)
    for i in range(blocks):
        x = layers.Conv2D(CSCNN_FILTERS[i], 3, padding="same", use_bias=not batchnorm,
                          name=f"conv{i + 1}")(x)
        if batchnorm:
            x = layers.BatchNormalization(momentum=0.9, name=f"bn{i + 1}")(x)
        x = layers.Activation("relu", name=f"relu{i + 1}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i + 1}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(classes, activation="softmax", name="classifier", dtype="float32")(x)
    return models.Model(inp, out, name="CSCNN")


def build_baseline(key, input_shape, classes, stretch=False):
    """ImageNet-initialised backbone, fully fine-tuned; GAP -> Dropout -> softmax."""
    mod_name, cls_name = BASELINES[key]
    apps = tf.keras.applications
    ctor = getattr(apps, cls_name, None) or getattr(getattr(apps, mod_name), cls_name)
    backbone = ctor(include_top=False, weights="imagenet", input_shape=input_shape)
    backbone.trainable = True
    inp = layers.Input(shape=input_shape, name="image_0_255")
    x = DecorrelationStretch(name="decorrelation_stretch")(inp) if stretch else inp
    x = BackbonePreprocess(mod_name, name="preprocess")(x)
    x = backbone(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(classes, activation="softmax", name="classifier", dtype="float32")(x)
    return models.Model(inp, out, name=cls_name)


def build_model(key, cfg, classes):
    shape = (cfg["img_size"], cfg["img_size"], 3)
    if key == "cscnn":
        return build_cscnn(shape, classes, cfg["blocks"], cfg["stretch"],
                           cfg["batchnorm"], cfg["dropout"])
    return build_baseline(key, shape, classes, cfg["stretch_baselines"])


# ============================================================== evaluation
def patient_bootstrap_ci(pats, y_true, y_pred, metric, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = {p: np.where(pats == p)[0] for p in np.unique(pats)}
    ids = list(groups)
    s = [metric(y_true[i], y_pred[i]) for i in
         (np.concatenate([groups[p] for p in rng.choice(ids, len(ids))]) for _ in range(n))]
    return [float(v) for v in np.percentile(s, [2.5, 97.5])]


def compute_metrics(test_df, y_true, probs, class_names):
    y_pred = probs.argmax(1)
    pats = test_df["patient_id"].values
    mf1 = lambda t, p: f1_score(t, p, average="macro")
    res = {}
    for k, fn in (("accuracy", accuracy_score), ("balanced_accuracy", balanced_accuracy_score),
                  ("macro_f1", mf1)):
        res[k] = float(fn(y_true, y_pred))
        res[k + "_ci95"] = patient_bootstrap_ci(pats, y_true, y_pred, fn)
    try:
        if len(class_names) == 2:
            res["macro_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            res["macro_auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr",
                                                   labels=range(len(class_names))))
    except ValueError:
        res["macro_auc"] = None
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    for i, c in enumerate(class_names):
        tp = cm[i, i]; fn_ = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn_ - fp
        res[f"sensitivity_{c}"] = float(tp / max(tp + fn_, 1))
        res[f"specificity_{c}"] = float(tn / max(tn + fp, 1))
    res["confusion_matrix"] = cm.tolist()
    res["report"] = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                          target_names=class_names, digits=3, zero_division=0)
    return res


# ================================================================ training
def make_generators(train, val, test, cfg, seed):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.2, height_shift_range=0.3,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                             fill_mode="reflect")          # outputs stay in 0-255
    plain = ImageDataGenerator()
    classes = sorted(train["label"].unique())
    kw = dict(x_col="filepath", y_col="label", classes=classes,
              target_size=(cfg["img_size"], cfg["img_size"]), class_mode="categorical",
              batch_size=cfg["batch"], interpolation="bilinear", seed=seed)
    return (aug.flow_from_dataframe(train, shuffle=True, **kw),
            plain.flow_from_dataframe(val, shuffle=False, **kw),
            plain.flow_from_dataframe(test, shuffle=False, **kw), classes)


def BalancedValCheckpoint(val_gen, wpath, patience=15, lr_patience=6):
    """Model selection on VALIDATION balanced accuracy (not val loss, which is dominated by
    the majority class when data are imbalanced). Saves the best weights, halves the
    learning rate after `lr_patience` epochs without improvement, stops after `patience`."""
    class _Cb(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.best, self.wait, self.lr_wait = -np.inf, 0, 0

        def on_epoch_end(self, epoch, logs=None):
            val_gen.reset()
            probs = self.model.predict(val_gen, verbose=0)
            y = np.asarray(val_gen.classes)
            bacc = balanced_accuracy_score(y, probs.argmax(1))
            score = bacc - 1e-3 * float(logs.get("val_loss", 0.0))   # tie-break on val loss
            logs["val_balanced_accuracy"] = bacc
            print(f"   val balanced accuracy: {bacc:.3f}")
            if score > self.best + 1e-4:
                self.best, self.wait, self.lr_wait = score, 0, 0
                self.model.save_weights(wpath)
            else:
                self.wait += 1
                self.lr_wait += 1
                if self.lr_wait >= lr_patience:
                    self.lr_wait = 0
                    try:
                        lr = self.model.optimizer.learning_rate
                        new = max(float(np.array(lr)) * 0.5, 1e-6)
                        lr.assign(new)
                        print(f"   learning rate -> {new:.2e}")
                    except Exception as e:
                        print("   (learning-rate reduction skipped:", e, ")")
                if self.wait >= patience:
                    self.model.stop_training = True
    return _Cb()


def export_layer_table(model, path):
    """Layer-by-layer architecture table (name, type, output shape, parameters)."""
    rows = []
    for l in model.layers:
        try:
            shape = tuple(l.output.shape)
        except Exception:
            shape = ""
        rows.append({"layer": l.name, "type": l.__class__.__name__,
                     "output_shape": str(shape), "parameters": int(l.count_params())})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"layer": "TOTAL", "type": "", "output_shape": "",
                       "parameters": int(model.count_params())}
    df.to_csv(path, index=False)


def run_one(key, cfg, fold, train, val, test, seed, outdir):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    os.makedirs(outdir, exist_ok=True)
    for n, d in (("train", train), ("val", val), ("test", test)):
        d.to_csv(os.path.join(outdir, f"split_{n}.csv"), index=False)
    split_summary(train, val, test).to_csv(os.path.join(outdir, "split_summary.csv"), index=False)

    tr, va, te, classes = make_generators(train, val, test, cfg, seed)
    y_tr = train["label"].map(tr.class_indices).values
    # square-root class weights: favour rare classes without collapsing onto them
    w = np.sqrt(compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_tr))
    cw = dict(enumerate(w / w.mean()))

    policy = tf.keras.mixed_precision.global_policy().name
    if key == "cscnn":                       # small model trained from scratch: full float32
        tf.keras.mixed_precision.set_global_policy("float32")
    model = build_model(key, cfg, len(classes))
    lr = cfg["lr"] or (1e-3 if key == "cscnn" else 1e-4)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    with open(os.path.join(outdir, "model_summary.txt"), "w", encoding="utf-8") as fh:
        model.summary(print_fn=lambda s, **k: fh.write(s + "\n"))
    export_layer_table(model, os.path.join(outdir, "layer_table.csv"))

    wpath = os.path.join(outdir, "best.weights.h5")
    cbs = [BalancedValCheckpoint(va, wpath, patience=cfg["patience"])]
    t0 = time.time()
    hist = model.fit(tr, validation_data=va, epochs=cfg["epochs"], class_weight=cw,
                     callbacks=cbs, verbose=2)
    train_time = time.time() - t0
    model.load_weights(wpath)
    pd.DataFrame(hist.history).to_csv(os.path.join(outdir, "history.csv"), index=False)

    # ---------------- test set: used exactly once
    model.predict(te, steps=1, verbose=0)                     # warm-up
    te.reset()
    t0 = time.time()
    probs = model.predict(te, verbose=0)
    ms_per_img = 1000 * (time.time() - t0) / len(probs)
    y_true = np.asarray(te.classes)
    m = compute_metrics(test.reset_index(drop=True), y_true, probs, classes)
    m.update(model=key, fold=fold, seed=seed, classes=classes,
             params=int(model.count_params()), ms_per_image=ms_per_img,
             train_time_s=train_time, epochs_run=len(hist.history["loss"]),
             best_epoch=int(np.argmax(hist.history.get("val_balanced_accuracy",
                                                     [-v for v in hist.history["val_loss"]]))) + 1, config=cfg)
    pred = test.reset_index(drop=True).copy()
    pred["y_true"] = [classes[i] for i in y_true]
    pred["y_pred"] = [classes[i] for i in probs.argmax(1)]
    for i, c in enumerate(classes):
        pred[f"p_{c}"] = probs[:, i]
    pred.to_csv(os.path.join(outdir, "test_predictions.csv"), index=False)
    with open(os.path.join(outdir, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    print(f"\n[{key} | seed {seed} | {fold}]\n{m['report']}")
    tf.keras.mixed_precision.set_global_policy(policy)
    return m


def model_tag(key, cfg):
    if key != "cscnn":
        return key + ("_stretch" if cfg["stretch_baselines"] else "")
    tag = f"cscnn_v3_{cfg['blocks']}blk"
    tag += "" if cfg["stretch"] else "_nostretch"
    tag += "" if cfg["batchnorm"] else "_nobn"
    tag += f"_do{cfg['dropout']}" if cfg["dropout"] else "_nodo"
    return tag


# pretty names -- edit to match your folder names
CLASS_NAMES = {"TONGUE-INFLM": "Chronic inflammatory",
               "TONGUE-NOEVD": "No evidence of malignancy",
               "TONGUE-MLDYSP": "Mild epithelial dysplasia",
               "TONGUE-MDSCC": "Moderately diff. SCC"}
CLASS_ABBR = {"TONGUE-INFLM": "INF", "TONGUE-NOEVD": "NEM",
              "TONGUE-MLDYSP": "MED", "TONGUE-MDSCC": "MD-SCC"}   # grid labels
MODEL_NAMES = {"vgg16": "VGG16", "vgg19": "VGG19", "resnet50": "ResNet50",
               "inceptionv3": "InceptionV3", "xception": "Xception",
               "mobilenetv2": "MobileNetV2", "resnet101": "ResNet101",
               "resnet50v2": "ResNet50V2", "inceptionresnetv2": "InceptionResNetV2",
               "densenet121": "DenseNet121", "densenet169": "DenseNet169",
               "mobilenetv3small": "MobileNetV3-Small", "mobilenetv3large": "MobileNetV3-Large",
               "efficientnetb0": "EfficientNetB0", "efficientnetv2b0": "EfficientNetV2-B0",
               "convnexttiny": "ConvNeXt-Tiny"}
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})


def cname(c):
    return CLASS_NAMES.get(c, c)


def mname(tag):
    if tag.startswith("cscnn"):
        return "CSCNN (proposed)" if tag == PROPOSED else "CSCNN " + tag.replace("cscnn_v3_", "")
    return MODEL_NAMES.get(tag, tag)


def save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ================================================================ loading
def load_results(root):
    out = {}
    for tag_dir in sorted(glob.glob(os.path.join(root, "*"))):
        runs = sorted(glob.glob(os.path.join(tag_dir, "*", "metrics.json")))
        if not runs:
            continue
        tag = os.path.basename(tag_dir)
        if tag.startswith("cscnn_") and not tag.startswith("cscnn_v3_"):
            continue                          # results of an earlier, superseded CSCNN version
        out[tag] = []
        for mpath in runs:
            rdir = os.path.dirname(mpath)
            with open(mpath) as fh:
                m = json.load(fh)
            m["run"] = os.path.basename(rdir)
            m["pred"] = pd.read_csv(os.path.join(rdir, "test_predictions.csv"))
            hp = os.path.join(rdir, "history.csv")
            m["history"] = pd.read_csv(hp) if os.path.exists(hp) else None
            m["dir"] = rdir
            out[tag].append(m)
    return out


def pooled(runs):
    """Concatenate test predictions of all runs of one model."""
    classes = runs[0]["classes"]
    df = pd.concat([r["pred"] for r in runs], ignore_index=True)
    y = df["y_true"].map({c: i for i, c in enumerate(classes)}).values
    p = df[[f"p_{c}" for c in classes]].values
    return classes, y, p


def msd(vals):
    vals = np.asarray([v for v in vals if v is not None], float)
    if len(vals) == 0:
        return np.nan, np.nan
    return vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0)


# ========================================================= per-model plots
def plot_confusion(tag, runs, out):
    classes, y, p = pooled(runs)
    cm = confusion_matrix(y, p.argmax(1), labels=range(len(classes)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    names = [cname(c) for c in classes]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title, fmt in ((axes[0], cm, "Counts", "d"), (axes[1], cmn, "Row-normalised", ".2f")):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=None if fmt == "d" else 1)
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Reference (pathology)")
        ax.set_title(f"{mname(tag)} - {title}")
        thr = mat.max() / 2
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > thr else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Test set, pooled over {len(runs)} run(s)", fontsize=9, y=0.02)
    save(fig, os.path.join(out, tag, "confusion_matrix"))
    return cmn, [CLASS_ABBR.get(c, c[:6]) for c in classes]


def roc_data(y, p, n):
    yb = label_binarize(y, classes=range(n))
    if n == 2:                                   # binary datasets (e.g. Rahman)
        yb = np.column_stack([1 - yb[:, 0], yb[:, 0]])
    fpr, tpr, aucs = {}, {}, {}
    for i in range(n):
        fpr[i], tpr[i], _ = roc_curve(yb[:, i], p[:, i])
        aucs[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(yb.ravel(), p.ravel())
    aucs["micro"] = auc(fpr["micro"], tpr["micro"])
    grid = np.linspace(0, 1, 501)
    mean_tpr = np.mean([np.interp(grid, fpr[i], tpr[i]) for i in range(n)], axis=0)
    fpr["macro"], tpr["macro"] = grid, mean_tpr
    aucs["macro"] = np.mean([aucs[i] for i in range(n)])
    return fpr, tpr, aucs


def plot_roc(tag, runs, out):
    classes, y, p = pooled(runs)
    n = len(classes)
    fpr, tpr, aucs = roc_data(y, p, n)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i in range(n):
        ax.plot(fpr[i], tpr[i], lw=1.6, label=f"{cname(classes[i])} (AUC = {aucs[i]:.3f})")
    ax.plot(fpr["micro"], tpr["micro"], "k:", lw=2, label=f"Micro-average (AUC = {aucs['micro']:.3f})")
    ax.plot(fpr["macro"], tpr["macro"], "k--", lw=2, label=f"Macro-average (AUC = {aucs['macro']:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("1 - Specificity (False positive rate)")
    ax.set_ylabel("Sensitivity (True positive rate)")
    ax.set_title(f"{mname(tag)}: one-vs-rest ROC (test set)")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, os.path.join(out, tag, "roc_curves"))
    return fpr["macro"], tpr["macro"], aucs


def plot_curves(tag, runs, out):
    hs = [r["history"] for r in runs if r["history"] is not None]
    if not hs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, (metric, ax) in enumerate(zip(("loss", "accuracy"), axes)):
        for j, h in enumerate(hs):
            lab = j == 0
            ax.plot(h[metric], color="C0", alpha=0.7, lw=1, label="Training" if lab else None)
            ax.plot(h["val_" + metric], color="C3", alpha=0.7, lw=1, label="Validation" if lab else None)
        ax.set_xlabel("Epoch"); ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{mname(tag)}: {metric} ({len(hs)} run(s))")
        ax.legend()
    save(fig, os.path.join(out, tag, "training_curves"))


# ======================================================= comparison plots
def comparison_table(res):
    rows, pc = [], []
    for tag, runs in res.items():
        row = {"model": mname(tag), "tag": tag, "runs": len(runs),
               "params": runs[0]["params"]}
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc"):
            m, s = msd([r.get(k) for r in runs])
            row[k + "_mean"], row[k + "_sd"] = m, s
            row[k] = f"{m:.3f} ± {s:.3f}"
        lo, hi = np.mean([r["balanced_accuracy_ci95"] for r in runs], axis=0)
        row["balanced_accuracy_ci95"] = f"{lo:.3f}–{hi:.3f}"
        row["ms_per_image"] = np.mean([r["ms_per_image"] for r in runs])
        rows.append(row)
        for c in runs[0]["classes"]:
            pc.append({"model": mname(tag), "class": cname(c),
                       "sensitivity": "%.3f ± %.3f" % msd([r[f"sensitivity_{c}"] for r in runs]),
                       "specificity": "%.3f ± %.3f" % msd([r[f"specificity_{c}"] for r in runs])})
    df = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    return df, pd.DataFrame(pc)


def plot_comparison(df, out):
    d = df.sort_values("balanced_accuracy_mean")
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.38 * len(d) + 1.5), sharey=True)
    colors = ["C3" if t == PROPOSED else "C0" for t in d["tag"]]
    for ax, k, lab in ((axes[0], "balanced_accuracy", "Balanced accuracy"), (axes[1], "macro_f1", "Macro-F1")):
        ax.barh(d["model"], d[k + "_mean"], xerr=d[k + "_sd"], color=colors, capsize=3)
        ax.set_xlabel(lab + " (mean ± SD)")
        lo = max(0, d[k + "_mean"].min() - 0.1)
        ax.set_xlim(lo, 1.0)
        for yv, (mv, sv) in enumerate(zip(d[k + "_mean"], d[k + "_sd"])):
            ax.text(min(mv + sv + 0.005, 0.99), yv, f"{mv:.3f}", va="center", fontsize=8)
    fig.suptitle("Test-set performance, patient-level split")
    save(fig, os.path.join(out, "Fig_model_comparison"))

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in df.iterrows():
        prop = r["tag"] == PROPOSED
        ax.errorbar(r["params"], r["macro_f1_mean"], yerr=r["macro_f1_sd"], fmt="o",
                    color="C3" if prop else "C0", ms=9 if prop else 6, capsize=3)
        ax.annotate(r["model"], (r["params"], r["macro_f1_mean"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters (log scale)"); ax.set_ylabel("Macro-F1 (mean ± SD)")
    ax.set_title("Performance vs model size")
    save(fig, os.path.join(out, "Fig_params_vs_performance"))


def plot_roc_all(rocs, out):
    fig, ax = plt.subplots(figsize=(7, 6))
    order = sorted(rocs, key=lambda t: -rocs[t][2]["macro"])
    for i, tag in enumerate(order):
        f, t, a = rocs[tag]
        prop = tag == PROPOSED
        ax.plot(f, t, lw=2.5 if prop else 1.1, color="C3" if prop else plt.cm.tab20(i % 20),
                label=f"{mname(tag)} ({a['macro']:.3f})", zorder=3 if prop else 2)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Macro-average ROC, all models (AUC in brackets)")
    ax.legend(fontsize=7, loc="lower right", ncol=1)
    save(fig, os.path.join(out, "Fig_roc_all_models"))


def plot_confusion_grid(cms, out):
    tags = list(cms)
    cols = 4
    rows = int(np.ceil(len(tags) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.1 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, tag in zip(axes, tags):
        m, names = cms[tag]
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        short = names
        ax.set_xticks(range(len(names)), short, rotation=45, fontsize=7)
        ax.set_yticks(range(len(names)), short, fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if m[i, j] > 0.5 else "black")
        ax.set_title(mname(tag), fontsize=9, color="C3" if tag == PROPOSED else "black")
    for ax in axes[len(tags):]:
        ax.axis("off")
    fig.suptitle("Row-normalised confusion matrices (rows: reference, columns: predicted)\n"
                 + ", ".join(f"{CLASS_ABBR.get(k, k)} = {v}" for k, v in CLASS_NAMES.items()), fontsize=10)
    fig.tight_layout()
    save(fig, os.path.join(out, "Fig_confusion_grid"))


def plot_ablation(df, out):
    d = df[df["tag"].str.startswith("cscnn")].sort_values("macro_f1_mean")
    if len(d) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(d) + 1.5))
    ax.barh(d["tag"], d["macro_f1_mean"], xerr=d["macro_f1_sd"], capsize=3,
            color=["C3" if t == PROPOSED else "C7" for t in d["tag"]])
    ax.set_xlabel("Macro-F1 (mean ± SD)"); ax.set_title("CSCNN ablation study")
    save(fig, os.path.join(out, "Fig_ablation"))


def mcnemar_table(res, out):
    """Exact McNemar test, CSCNN vs each model, on every shared run (same patients)."""
    rows = []
    base = {r["run"]: r["pred"] for r in res[PROPOSED]}
    for tag, runs in res.items():
        if tag == PROPOSED:
            continue
        for r in runs:
            if r["run"] not in base:
                continue
            a = base[r["run"]].set_index("filepath")
            b = r["pred"].set_index("filepath").loc[a.index]
            ca, cb = a["y_true"] == a["y_pred"], b["y_true"] == b["y_pred"]
            n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            rows.append({"comparison": f"CSCNN vs {mname(tag)}", "run": r["run"],
                         "cscnn_right_other_wrong": n01, "cscnn_wrong_other_right": n10,
                         "p_value": p})
    df = pd.DataFrame(rows)
    if len(df):
        df["significant_0.05_bonferroni"] = df["p_value"] < 0.05 / df["comparison"].nunique()
        df.to_csv(os.path.join(out, "Table_mcnemar.csv"), index=False)


# =============================================== stretch illustration
def plot_stretch_examples(meta, out, n_per_class=1, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    classes = sorted(meta["label"].unique())
    fig, axes = plt.subplots(len(classes) * n_per_class, 3,
                             figsize=(10, 3.2 * len(classes) * n_per_class))
    axes = np.atleast_2d(axes)
    r = 0
    for c in classes:
        paths = meta.loc[meta["label"] == c, "filepath"].values
        for pth in rng.choice(paths, min(n_per_class, len(paths)), replace=False):
            img = np.asarray(Image.open(pth).convert("RGB"), dtype=np.float32)
            st = decorrelation_stretch_np(img)
            c0 = np.corrcoef(img.reshape(-1, 3).T)
            c1 = np.corrcoef(st.reshape(-1, 3).T)
            axes[r, 0].imshow(img.astype(np.uint8)); axes[r, 0].set_title(f"{cname(c)}: original", fontsize=9)
            axes[r, 1].imshow(st.astype(np.uint8)); axes[r, 1].set_title("Decorrelation stretched", fontsize=9)
            for ch, col in enumerate("rgb"):
                axes[r, 2].hist(img[..., ch].ravel(), bins=64, color=col, alpha=0.35, histtype="stepfilled")
                axes[r, 2].hist(st[..., ch].ravel(), bins=64, color=col, histtype="step", lw=1.3)
            axes[r, 2].set_title(f"RGB histograms (filled: original, line: stretched)\n"
                                 f"mean |r| between channels: {np.abs(c0[np.triu_indices(3, 1)]).mean():.2f}"
                                 f" -> {np.abs(c1[np.triu_indices(3, 1)]).mean():.2f}", fontsize=8)
            axes[r, 0].axis("off"); axes[r, 1].axis("off")
            r += 1
    fig.tight_layout()
    save(fig, os.path.join(out, "Fig_stretch_examples"))


# ====================================================== Grad-CAM & t-SNE
def _load_cscnn(run):
    import tensorflow as tf
    cfg = run["config"]
    m = build_cscnn((cfg["img_size"], cfg["img_size"], 3), len(run["classes"]),
                    cfg["blocks"], cfg["stretch"], cfg["batchnorm"], cfg["dropout"])
    m.load_weights(os.path.join(run["dir"], "best.weights.h5"))
    return m, cfg, tf


def _load_imgs(paths, size):
    from PIL import Image
    return np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR),
                                dtype=np.float32) for p in paths])


def plot_gradcam(tag, runs, out, per_class=3):
    run = runs[0]
    model, cfg, tf = _load_cscnn(run)
    last = f"relu{cfg['blocks']}"
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(last).output, model.output])
    pred = run["pred"]
    classes = run["classes"]
    picks = []
    for c in classes:
        sub = pred[pred["y_true"] == c]
        ok, bad = sub[sub["y_pred"] == c], sub[sub["y_pred"] != c]
        picks += [(r, True) for _, r in ok.head(per_class - 1).iterrows()]
        picks += [(r, False) for _, r in bad.head(1).iterrows()] or \
                 [(r, True) for _, r in ok.iloc[per_class - 1:per_class].iterrows()]
    imgs = _load_imgs([p["filepath"] for p, _ in picks], cfg["img_size"])
    cols = per_class * 2
    rows = int(np.ceil(len(picks) / per_class))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    axes = np.atleast_2d(axes)
    for k, ((row, correct), img) in enumerate(zip(picks, imgs)):
        with tf.GradientTape() as tape:
            conv, preds = grad_model(img[None])
            idx = int(tf.argmax(preds[0]))
            score = preds[:, idx]
        g = tape.gradient(score, conv)[0]
        w = tf.reduce_mean(g, axis=(0, 1))
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * w, axis=-1)).numpy()
        cam = cam / (cam.max() + 1e-8)
        cam = tf.image.resize(cam[..., None], img.shape[:2]).numpy()[..., 0]
        rr, cc = divmod(k, per_class)
        a0, a1 = axes[rr, 2 * cc], axes[rr, 2 * cc + 1]
        a0.imshow(img.astype(np.uint8)); a0.axis("off")
        a1.imshow(img.astype(np.uint8)); a1.imshow(cam, cmap="jet", alpha=0.45); a1.axis("off")
        a0.set_title(f"True: {cname(row['y_true'])[:18]}", fontsize=7)
        a1.set_title(f"Pred: {cname(classes[idx])[:18]}", fontsize=7,
                     color="green" if correct else "red")
    for ax in axes.ravel()[2 * len(picks):]:
        ax.axis("off")
    fig.suptitle(f"Grad-CAM ({mname(tag)}, layer {last}); red title = misclassified", fontsize=10)
    fig.tight_layout()
    save(fig, os.path.join(out, f"Fig_gradcam_{tag}"))


def plot_tsne(tag, runs, out, max_n=2000):
    from sklearn.manifold import TSNE
    run = runs[0]
    model, cfg, tf = _load_cscnn(run)
    feat_model = tf.keras.Model(model.inputs, model.get_layer("gap").output)
    pred = run["pred"].head(max_n)
    feats = []
    for i in range(0, len(pred), 64):
        feats.append(feat_model.predict(_load_imgs(pred["filepath"].values[i:i + 64], cfg["img_size"]),
                                        verbose=0))
    emb = TSNE(n_components=2, perplexity=min(30, max(5, len(pred) // 10)),
               init="pca", random_state=0).fit_transform(np.concatenate(feats))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for c in run["classes"]:
        m = pred["y_true"].values == c
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.7, label=cname(c))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t-SNE of {mname(tag)} features (test set)")
    ax.legend(markerscale=2, fontsize=8)
    save(fig, os.path.join(out, f"Fig_tsne_{tag}"))


# =====================================================================
#            IMAGE ENHANCEMENT EVALUATION (decorrelation stretch vs others)
# =====================================================================
ENH_METHODS = ["Original", "Linear stretch", "Histogram eq.", "CLAHE",
               "Reinhard", "Macenko", "Decorrelation stretch"]
PROPOSED_ENH = "Decorrelation stretch"
# metric: (column, label, direction)  direction +1 = higher is better, -1 = lower is better
ENH_METRICS = [("rms_contrast", "RMS contrast", +1),
               ("entropy", "Entropy (bits)", +1),
               ("avg_gradient", "Average gradient (edge strength)", +1),
               ("colourfulness", "Colourfulness", +1),
               ("channel_corr", "Mean inter-channel |r| (lower = more decorrelated)", -1),
               ("ssim", "SSIM to original (structure preservation)", +1),
               ("brisque", "BRISQUE (lower = better perceptual quality)", -1)]


def _u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def enh_linear(img):
    """Per-channel linear contrast stretch between the 1st and 99th percentiles."""
    out = np.empty_like(img, dtype=np.float32)
    for c in range(3):
        lo, hi = np.percentile(img[..., c], (1, 99))
        out[..., c] = (img[..., c] - lo) * 255.0 / max(hi - lo, 1e-6)
    return np.clip(out, 0, 255)


def enh_hist_eq(img):
    import cv2
    ycc = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB).astype(np.float32)


def enh_clahe(img):
    import cv2
    lab = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)


def enh_reinhard(img, target):
    """Reinhard colour transfer in LAB to the dataset-average LAB statistics."""
    import cv2
    lab = cv2.cvtColor((np.clip(img, 0, 255) / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB)
    m, s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    lab = (lab - m) / s * target[1] + target[0]
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB) * 255, 0, 255)


def enh_macenko(img, Io=240, alpha=1, beta=0.15):
    """Macenko et al. (2009) stain normalisation to the standard H&E reference."""
    he_ref = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
    max_c_ref = np.array([1.9705, 1.0308])
    h, w, _ = img.shape
    od = -np.log((img.reshape(-1, 3).astype(np.float64) + 1) / Io)
    odhat = od[~np.any(od < beta, axis=1)]
    if len(odhat) < 100:
        raise ValueError("too little stained tissue")
    _, ev = np.linalg.eigh(np.cov(odhat.T))
    t = odhat @ ev[:, 1:3]
    phi = np.arctan2(t[:, 1], t[:, 0])
    lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    v1 = ev[:, 1:3] @ np.array([np.cos(lo), np.sin(lo)])
    v2 = ev[:, 1:3] @ np.array([np.cos(hi), np.sin(hi)])
    he = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T
    c = np.linalg.lstsq(he, od.T, rcond=None)[0]
    max_c = np.array([np.percentile(c[0], 99), np.percentile(c[1], 99)])
    c *= (max_c_ref / np.maximum(max_c, 1e-6))[:, None]
    return np.clip((Io * np.exp(-he_ref @ c)).T.reshape(h, w, 3), 0, 255).astype(np.float32)


def apply_enhancement(name, img, reinhard_target):
    if name == "Original":
        return img
    if name == "Linear stretch":
        return enh_linear(img)
    if name == "Histogram eq.":
        return enh_hist_eq(img)
    if name == "CLAHE":
        return enh_clahe(img)
    if name == "Reinhard":
        return enh_reinhard(img, reinhard_target)
    if name == "Macenko":
        return enh_macenko(img)
    if name == "Decorrelation stretch":
        return decorrelation_stretch_np(img)
    raise ValueError(name)


def _brisque_fn():
    """Returns a BRISQUE scorer (piq) or None if unavailable."""
    try:
        import torch
        try:
            import piq
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "piq"], check=True)
            import piq
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        def score(u8):
            x = torch.from_numpy(u8).permute(2, 0, 1)[None].float().div(255).to(dev)
            return float(piq.brisque(x, data_range=1.0))
        return score
    except Exception as e:
        print("[WARN] BRISQUE unavailable, skipped:", e)
        return None


def image_metrics(u8, ref_gray, brisque=None):
    import cv2
    from skimage.metrics import structural_similarity
    g = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    gf = g.astype(np.float64)
    p = np.bincount(g.ravel(), minlength=256) / g.size
    p = p[p > 0]
    rgb = u8.reshape(-1, 3).astype(np.float64)
    rg = rgb[:, 0] - rgb[:, 1]
    yb = 0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.nan_to_num(np.corrcoef(rgb.T))
    gx = cv2.Sobel(gf, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_64F, 0, 1, ksize=3)
    return {"rms_contrast": gf.std() / 255.0,
            "entropy": float(-(p * np.log2(p)).sum()),
            "avg_gradient": float(np.sqrt(gx ** 2 + gy ** 2).mean() / 255.0),
            "colourfulness": float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())),
            "channel_corr": float(np.abs(cc[np.triu_indices(3, 1)]).mean()),
            "ssim": float(structural_similarity(ref_gray, g, data_range=255)),
            "brisque": brisque(u8) if brisque else np.nan}


def _load_eval_img(path, max_side):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def reinhard_target(meta, max_side, n=100, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    paths = rng.choice(meta["filepath"].values, min(n, len(meta)), replace=False)
    stats = []
    for p in paths:
        lab = cv2.cvtColor(_load_eval_img(p, max_side) / 255.0, cv2.COLOR_RGB2LAB).reshape(-1, 3)
        stats.append((lab.mean(0), lab.std(0)))
    return np.mean([s[0] for s in stats], 0), np.mean([s[1] for s in stats], 0)


def compute_enhancement_metrics(meta, out_dir, max_side):
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, "enh_metrics_per_image.csv")
    done = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
    done_paths = set(done["filepath"]) if len(done) else set()
    todo = meta[~meta["filepath"].isin(done_paths)]
    if len(todo) == 0:
        return done
    print(f"[INFO] enhancement metrics for {len(todo)} images x {len(ENH_METHODS)} methods ...")
    import cv2
    target = reinhard_target(meta, max_side)
    brisque = _brisque_fn() if ENH_BRISQUE else None
    rows, t0 = [], time.time()
    for k, r in enumerate(todo.itertuples(index=False), 1):
        img = _load_eval_img(r.filepath, max_side)
        ref_gray = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2GRAY)
        for mth in ENH_METHODS:
            try:
                u8 = _u8(apply_enhancement(mth, img, target))
                m = image_metrics(u8, ref_gray, brisque)
                ok = True
            except Exception:
                m, ok = {c: np.nan for c, _, _ in ENH_METRICS}, False
            rows.append({"filepath": r.filepath, "label": r.label,
                         "magnification": getattr(r, "magnification", ""), "method": mth,
                         "ok": ok, **m})
        if k % 50 == 0 or k == len(todo):
            print(f"   {k}/{len(todo)} images ({time.time() - t0:.0f}s)")
            pd.concat([done, pd.DataFrame(rows)]).to_csv(csv, index=False)   # checkpoint
    df = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(csv, index=False)
    return df


def _holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(p) - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def enhancement_statistics(df, out_dir):
    from scipy.stats import friedmanchisquare, wilcoxon
    metrics = [(c, l, d) for c, l, d in ENH_METRICS if df[c].notna().any()]
    q = lambda s: f"{s.median():.4f} [{s.quantile(.25):.4f}–{s.quantile(.75):.4f}]"
    # summary: median [IQR]
    summ = pd.DataFrame({l: df.groupby("method")[c].apply(q) for c, l, _ in metrics}).reindex(ENH_METHODS)
    summ.to_csv(os.path.join(out_dir, "Table_enh_summary.csv"))
    byc = df.groupby(["label", "method"])[[c for c, _, _ in metrics]].median().reset_index()
    byc["label"] = byc["label"].map(cname)
    byc.to_csv(os.path.join(out_dir, "Table_enh_by_class.csv"), index=False)
    if df["magnification"].fillna("").astype(str).str.strip().ne("").any():
        df.groupby(["magnification", "method"])[[c for c, _, _ in metrics]].median().to_csv(
            os.path.join(out_dir, "Table_enh_by_magnification.csv"))
    # paired tests (same images under each method)
    wide = {c: df.pivot_table(index="filepath", columns="method", values=c) for c, _, _ in metrics}
    fr, wr = [], []
    for c, l, d in metrics:
        w = wide[c].dropna()
        if len(w) > 5:
            stat, p = friedmanchisquare(*[w[m] for m in ENH_METHODS if m in w])
            fr.append({"metric": l, "n_images": len(w), "friedman_chi2": stat, "p_value": p})
        comps = [m for m in ENH_METHODS if m != PROPOSED_ENH and m in w]
        ps, recs = [], []
        for m in comps:
            diff = w[PROPOSED_ENH] - w[m]
            try:
                p = wilcoxon(w[PROPOSED_ENH], w[m]).pvalue
            except ValueError:
                p = 1.0
            ps.append(p)
            better = (diff.median() * d) > 0
            recs.append({"metric": l, "comparison": f"{PROPOSED_ENH} vs {m}",
                         "median_difference": diff.median(), "p_value": p,
                         "decorrelation_better": bool(better)})
        for rec, pa in zip(recs, _holm(ps)):
            rec["p_holm"] = pa
            rec["significant_0.05"] = pa < 0.05
        wr += recs
    pd.DataFrame(fr).to_csv(os.path.join(out_dir, "Table_enh_friedman.csv"), index=False)
    pd.DataFrame(wr).to_csv(os.path.join(out_dir, "Table_enh_wilcoxon.csv"), index=False)
    # mean rank of each method per metric (1 = best)
    ranks = {}
    for c, l, d in metrics:
        w = wide[c].dropna()
        ranks[l.split(" (")[0]] = (-d * w).rank(axis=1).mean()
    rk = pd.DataFrame(ranks).reindex(ENH_METHODS)
    rk.to_csv(os.path.join(out_dir, "Table_enh_mean_ranks.csv"))
    print("\n[ENHANCEMENT] median [IQR] per method:\n", summ.to_string())
    return summ, rk


def enhancement_figures(df, meta, rk, out_dir, max_side):
    metrics = [(c, l, d) for c, l, d in ENH_METRICS if df[c].notna().any()]
    target = reinhard_target(meta, max_side)
    # 1. visual comparison grid: one image per class x all methods (centre crop)
    classes = sorted(meta["label"].unique())
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(len(classes), len(ENH_METHODS),
                             figsize=(2.3 * len(ENH_METHODS), 2.4 * len(classes)))
    axes = np.atleast_2d(axes)
    for i, c in enumerate(classes):
        img = _load_eval_img(rng.choice(meta.loc[meta["label"] == c, "filepath"].values), max_side)
        h, w = img.shape[:2]
        s = min(h, w, 512)
        img = img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        for j, m in enumerate(ENH_METHODS):
            ax = axes[i, j]
            try:
                ax.imshow(_u8(apply_enhancement(m, img, target)))
            except Exception:
                ax.text(0.5, 0.5, "n/a", ha="center")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(m, fontsize=9, color="C3" if m == PROPOSED_ENH else "black")
            if j == 0:
                ax.set_ylabel(cname(c), fontsize=8)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "Fig_enh_visual_comparison"))
    # 2. box plots per metric
    n = len(metrics)
    cols = 3
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(5.2 * cols, 3.6 * np.ceil(n / cols)))
    axes = np.atleast_1d(axes).ravel()
    for ax, (c, l, d) in zip(axes, metrics):
        data = [df.loc[df["method"] == m, c].dropna().values for m in ENH_METHODS]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, m in zip(bp["boxes"], ENH_METHODS):
            patch.set_facecolor("C3" if m == PROPOSED_ENH else "C0"); patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(ENH_METHODS) + 1), ENH_METHODS, rotation=40, ha="right", fontsize=8)
        ax.set_title(l, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "Fig_enh_metric_boxplots"))
    # 3. mean-rank heat-map
    fig, ax = plt.subplots(figsize=(1.3 * rk.shape[1] + 2, 0.5 * rk.shape[0] + 1.5))
    im = ax.imshow(rk.values, cmap="RdYlGn_r")
    ax.set_xticks(range(rk.shape[1]), rk.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(rk.shape[0]), rk.index)
    for i in range(rk.shape[0]):
        for j in range(rk.shape[1]):
            ax.text(j, i, f"{rk.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean rank (1 = best)")
    ax.set_title("Mean rank of each method per metric (paired, per image)")
    save(fig, os.path.join(out_dir, "Fig_enh_rank_heatmap"))
    # 4. per-class medians for key metrics
    keys = [m for m in metrics if m[0] in ("rms_contrast", "entropy", "avg_gradient")]
    if keys:
        fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4))
        axes = np.atleast_1d(axes)
        med = df.groupby(["label", "method"])[[k[0] for k in keys]].median()
        x = np.arange(len(classes)); bw = 0.8 / len(ENH_METHODS)
        for ax, (c, l, _) in zip(axes, keys):
            for j, m in enumerate(ENH_METHODS):
                vals = [med.loc[(cl, m), c] if (cl, m) in med.index else np.nan for cl in classes]
                ax.bar(x + j * bw - 0.4 + bw / 2, vals, bw, label=m,
                       color="C3" if m == PROPOSED_ENH else plt.cm.Blues(0.3 + 0.1 * j))
            ax.set_xticks(x, [CLASS_ABBR.get(cl, cl) for cl in classes]); ax.set_title(l, fontsize=9)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        save(fig, os.path.join(out_dir, "Fig_enh_by_class"))
    # 5. RGB pixel scatter before / after decorrelation stretch
    img = _load_eval_img(meta["filepath"].iloc[0], max_side)
    st = decorrelation_stretch_np(img)
    idx = np.random.default_rng(0).choice(img.shape[0] * img.shape[1], min(5000, img.shape[0] * img.shape[1]), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, im_, t in ((axes[0], img, "Original"), (axes[1], st, "Decorrelation stretched")):
        px = im_.reshape(-1, 3)[idx]
        r = np.corrcoef(px.T)
        ax.scatter(px[:, 0], px[:, 1], s=2, alpha=0.3, label=f"R–G (r = {r[0, 1]:.2f})")
        ax.scatter(px[:, 0], px[:, 2], s=2, alpha=0.3, label=f"R–B (r = {r[0, 2]:.2f})")
        ax.set_xlim(0, 255); ax.set_ylim(0, 255)
        ax.set_xlabel("Red"); ax.set_ylabel("Green / Blue"); ax.set_title(t)
        ax.legend(markerscale=5, fontsize=8)
    save(fig, os.path.join(out_dir, "Fig_enh_channel_scatter"))


def run_enhancement(meta):
    out = os.path.join(WORK_DIR, "enhancement")
    df = compute_enhancement_metrics(meta, out, ENH_MAX_SIDE)
    df = df[df["ok"].astype(bool)]
    _, rk = enhancement_statistics(df, out)
    enhancement_figures(df, meta, rk, out, ENH_MAX_SIDE)
    print(f"[DONE] enhancement tables and figures in {out}")


# =====================================================================
#            FAST TRAINING CACHE (images pre-resized once per session)
# =====================================================================
def build_cache(meta, side=None):
    """Write a resized copy of every image to the Colab disk (longest side = CACHE_SIDE).
    Decoding large microscope JPEGs every epoch is the main bottleneck; this removes it.
    Returns metadata whose 'filepath' points to the cached files ('orig_filepath' kept)."""
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    side = side or CACHE_SIDE
    root = f"/content/cache_{DATASET}_{side}" if os.path.isdir("/content") else f"/tmp/cache_{DATASET}_{side}"
    meta = meta.copy()
    meta["orig_filepath"] = meta["filepath"]
    meta["filepath"] = [os.path.join(root, f"{i:06d}.png") for i in range(len(meta))]
    todo = [(s, d) for s, d in zip(meta["orig_filepath"], meta["filepath"]) if not os.path.exists(d)]
    if todo:
        os.makedirs(root, exist_ok=True)
        print(f"[INFO] caching {len(todo)} images at {side}px for fast training ...")

        def work(sd):
            im = Image.open(sd[0]).convert("RGB")
            im.thumbnail((side, side), Image.BILINEAR)
            im.save(sd[1])
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(work, todo))
    return meta


# =====================================================================
#     AUGMENTATION-ORDER EXPERIMENT: augment-then-split vs split-then-augment
# =====================================================================
def build_legacy_shallownet(classes, size=32):
    """Four-block baseline network: Conv(32, 3x3, same) ->
    MaxPool(2) -> ReLU blocks, Flatten, Dense softmax (29,156 parameters at 32x32x3)."""
    inp = layers.Input(shape=(size, size, 3))
    x = inp
    for i in range(4):
        x = layers.Conv2D(32, 3, padding="same", name=f"conv{i + 1}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i + 1}")(x)
        x = layers.Activation("relu", name=f"relu{i + 1}")(x)
    x = layers.Flatten()(x)
    out = layers.Dense(classes, activation="softmax", dtype="float32")(x)
    return models.Model(inp, out, name="LegacyShallowNet")


def _augment_pool(meta, target_per_class=1300, size=32, seed=0):
    """Offline class balancing: every class is topped up
    with augmented copies until it has ~target_per_class images. Each copy remembers
    its source image (column 'source')."""
    import cv2
    from PIL import Image
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    gen = ImageDataGenerator(rotation_range=25, width_shift_range=0.2, height_shift_range=0.3,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                             fill_mode="nearest")
    rng = np.random.default_rng(seed)
    X, rows = [], []
    small = lambda a: cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
    for label, grp in meta.groupby("label"):
        n_copies = max(int(np.ceil(target_per_class / len(grp))) - 1, 0)
        for r in grp.itertuples(index=False):
            img = np.asarray(Image.open(r.filepath).convert("RGB"), dtype=np.float32)
            X.append(small(img)); rows.append({"label": label, "source": r.patient_id, "augmented": False})
            for _ in range(n_copies):
                t = gen.random_transform(img, seed=int(rng.integers(1 << 30)))
                X.append(small(t)); rows.append({"label": label, "source": r.patient_id, "augmented": True})
    df = pd.DataFrame(rows)
    df["patient_id"] = df["source"]
    return np.stack(X).astype(np.float32) / 255.0, df


def _fit_legacy(Xtr, ytr, Xval, yval, n_cls, epochs, seed, early_stop):
    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(seed)
    m = build_legacy_shallownet(n_cls, Xtr.shape[1])
    m.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.06),
              loss="categorical_crossentropy", metrics=["accuracy"])
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                            restore_best_weights=True)] if early_stop else []
    h = m.fit(Xtr, tf.keras.utils.to_categorical(ytr, n_cls), batch_size=32, epochs=epochs,
              validation_data=(Xval, tf.keras.utils.to_categorical(yval, n_cls)),
              callbacks=cbs, verbose=0)
    return m, h


def run_leakage_demo(meta):
    """(A) Legacy protocol: offline augmentation to balance the
    classes, THEN a random 80/20 split of all images (originals + copies), test set also
    used as validation, 100 epochs.  (B) CORRECTED protocol, same model, same augmentation:
    split ORIGINAL images first (group-aware 5-fold CV), augmented copies only of training
    images, separate validation set, test = untouched originals.
    The difference between A and B is the accuracy inflation caused by leakage."""
    from sklearn.model_selection import train_test_split
    out = os.path.join(WORK_DIR, "leakage_demo_v3")
    summ_csv = os.path.join(out, "Table_leakage_demo.csv")
    if os.path.exists(summ_csv):
        print("[SKIP] leakage demo already done"); return
    os.makedirs(out, exist_ok=True)
    epochs = 2 if QUICK_TEST else 100
    runs = 1 if QUICK_TEST else 5
    print("[INFO] leakage demo: building the offline-augmented pool")
    X, pool = _augment_pool(meta)
    classes = sorted(pool["label"].unique())
    y = pool["label"].map({c: i for i, c in enumerate(classes)}).values
    rows = []
    # ---- (A) legacy protocol
    for r in range(runs):
        idx = np.arange(len(pool))
        tr, te = train_test_split(idx, test_size=0.20, random_state=50 + r, stratify=y)
        m, _ = _fit_legacy(X[tr], y[tr], X[te], y[te], len(classes), epochs, 50 + r, early_stop=False)
        probs = m.predict(X[te], verbose=0)
        met = compute_metrics(pool.iloc[te].reset_index(drop=True), y[te], probs, classes)
        leak = float(pool.iloc[te]["source"].isin(set(pool.iloc[tr]["source"])).mean())
        rows.append({"protocol": "Legacy (augment, then random split; test = validation)",
                     "run": r + 1, "test_images": len(te), "test_from_sources_seen_in_training": leak,
                     **{k: met[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")}})
        print(f"   legacy run {r + 1}: acc {met['accuracy']:.3f}, leaked test fraction {leak:.2f}")
    # ---- (B) corrected protocol (same model, same augmentation)
    orig = pool[~pool["augmented"]]
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
    for k, (trv, tst) in enumerate(folds.split(orig, orig["label"], orig["source"])):
        if k >= runs:
            break
        test_src = set(orig.iloc[tst]["source"])
        trval_orig = orig.iloc[trv]
        inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=1)
        a, b = next(inner.split(trval_orig, trval_orig["label"], trval_orig["source"]))
        val_src = set(trval_orig.iloc[b]["source"])
        is_test = pool["source"].isin(test_src) & ~pool["augmented"]
        is_val = pool["source"].isin(val_src) & ~pool["augmented"]
        is_tr = ~pool["source"].isin(test_src | val_src)          # originals + their copies
        tr, va, te = (np.where(v)[0] for v in (is_tr, is_val, is_test))
        assert not set(pool.iloc[te]["source"]) & set(pool.iloc[tr]["source"])
        m, _ = _fit_legacy(X[tr], y[tr], X[va], y[va], len(classes), epochs, 1 + k, early_stop=True)
        probs = m.predict(X[te], verbose=0)
        met = compute_metrics(pool.iloc[te].reset_index(drop=True), y[te], probs, classes)
        rows.append({"protocol": "Corrected (split originals first; augment training only)",
                     "run": k + 1, "test_images": len(te), "test_from_sources_seen_in_training": 0.0,
                     **{kk: met[kk] for kk in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")}})
        print(f"   corrected fold {k + 1}: acc {met['accuracy']:.3f}, bal acc {met['balanced_accuracy']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "leakage_demo_all_runs.csv"), index=False)
    summ = df.groupby("protocol").agg(runs=("run", "count"), test_images=("test_images", "mean"),
                                      leaked_test_fraction=("test_from_sources_seen_in_training", "mean"),
                                      accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
                                      balanced_accuracy_mean=("balanced_accuracy", "mean"),
                                      balanced_accuracy_sd=("balanced_accuracy", "std"),
                                      macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                                      macro_auc_mean=("macro_auc", "mean")).reset_index()
    summ.to_csv(summ_csv, index=False)
    print(summ.to_string(index=False))
    # ---- figure
    fig, ax = plt.subplots(figsize=(7.5, 4))
    metrics_ = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1")]
    x = np.arange(len(metrics_))
    for j, (prot, col) in enumerate(zip(summ["protocol"], ("C7", "C3"))):
        vals = [summ.loc[summ["protocol"] == prot, f"{m}_mean"].item() for m, _ in metrics_]
        sds = [np.nan_to_num(summ.loc[summ["protocol"] == prot, f"{m}_sd"].item()) for m, _ in metrics_]
        ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, yerr=sds, capsize=3, color=col,
               label=prot.split(" (")[0] + " protocol")
        for xi, v in zip(x + (j - 0.5) * 0.38, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [l for _, l in metrics_]); ax.set_ylim(0, 1.1)
    ax.set_ylabel("Test performance (mean ± SD)")
    ax.set_title("Same model, same images: effect of the evaluation protocol")
    ax.legend(fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    save(fig, os.path.join(out, "Fig_leakage_demo"))
    if MIXED_PRECISION and tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


# =====================================================================
#                       PER-MAGNIFICATION RESULTS
# =====================================================================
def per_magnification_table(res, out):
    rows = []
    for tag, runs in res.items():
        df = pd.concat([r["pred"] for r in runs], ignore_index=True)
        if "magnification" not in df or df["magnification"].fillna("").astype(str).str.strip().eq("").all():
            continue
        for mag, g in df.groupby(df["magnification"].fillna("unknown")):
            rows.append({"model": mname(tag), "magnification": mag, "test_images": len(g),
                         "accuracy": accuracy_score(g["y_true"], g["y_pred"]),
                         "balanced_accuracy": balanced_accuracy_score(g["y_true"], g["y_pred"])})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out, "Table_per_magnification.csv"), index=False)

CLASS_NAMES.update(CLASS_NAMES_OVERRIDE)
CLASS_ABBR.update(CLASS_ABBR_OVERRIDE)


# =====================================================================
#                                 DRIVER
# =====================================================================
def prepare_ndbufes():
    """Unzip NDB-UFES images to the Colab disk and build metadata from ndb-ufes.csv
    (label = TaskIV: OSCC / leukoplakia with dysplasia / leukoplakia without dysplasia;
    grouping = the dataset's own patient_id -> genuine patient-level split)."""
    import zipfile
    img_dir = os.path.join(LOCAL_COPY, "images")
    if not os.path.isdir(img_dir):
        print("[INFO] unzipping NDB-UFES images (once per session)...")
        with zipfile.ZipFile(os.path.join(DATASET_DIR, "images.zip")) as z:
            z.extractall(LOCAL_COPY)
    info = pd.read_csv(os.path.join(DATASET_DIR, "ndb-ufes.csv"))
    meta = pd.DataFrame({"filepath": [os.path.join(img_dir, p) for p in info["path"]],
                         "label": info["TaskIV"].astype(str).str.strip(),
                         "patient_id": "pt_" + info["patient_id"].astype(str),
                         "slide_id": "pt_" + info["patient_id"].astype(str),   # lesions nest in patients
                         "magnification": ""})
    missing = ~meta["filepath"].map(os.path.exists)
    if missing.any():
        print(f"[WARN] {missing.sum()} images listed in ndb-ufes.csv not found -> skipped")
        meta = meta[~missing]
    os.makedirs(WORK_DIR, exist_ok=True)
    meta.to_csv(METADATA_CSV, index=False)
    meta = load_metadata(METADATA_CSV)
    print(f"[INFO] NDB-UFES: {len(meta)} images, {meta['patient_id'].nunique()} patients")
    print(meta.groupby("label").agg(images=("filepath", "count"),
                                    patients=("patient_id", "nunique")).to_string())
    return meta


def mount_drive():
    try:
        from google.colab import drive
        if not os.path.isdir("/content/drive/MyDrive"):
            drive.mount("/content/drive")
    except ImportError:
        print("[INFO] not running in Colab; using local paths as given")


def prepare_metadata(require_ids=True):
    """Create metadata.csv on first run; afterwards load and validate it."""
    ds, wk = os.path.normpath(DATASET_DIR), os.path.normpath(WORK_DIR)
    if wk == ds or wk.startswith(ds + os.sep):
        sys.exit("[ERROR] WORK_DIR must not be inside DATASET_DIR (figures would be read as images).")
    os.makedirs(WORK_DIR, exist_ok=True)
    if os.path.exists(METADATA_CSV):           # rebuild if it was made from another folder
        old = pd.read_csv(METADATA_CSV, dtype=str)
        if not old["filepath"].str.startswith(DATASET_DIR).all():
            print("[INFO] metadata.csv belongs to a different dataset folder -> rebuilding")
            os.rename(METADATA_CSV, METADATA_CSV + ".old")
    if not os.path.exists(METADATA_CSV):
        if not os.path.isdir(DATASET_DIR):
            sys.exit(f"[ERROR] dataset folder not found: {DATASET_DIR}")
        make_metadata(DATASET_DIR, METADATA_CSV)
    meta = pd.read_csv(METADATA_CSV, dtype=str).fillna("")
    names = meta["filepath"].map(os.path.basename)
    for col, rx in (("patient_id", PATIENT_ID_REGEX), ("slide_id", SLIDE_ID_REGEX),
                    ("magnification", MAGNIFICATION_REGEX)):
        if rx and (meta[col] == "").any():
            meta[col] = names.str.extract(rx, flags=re.IGNORECASE)[0].fillna("")
    meta.to_csv(METADATA_CSV, index=False)
    missing = (meta["patient_id"].str.strip() == "").sum()
    if missing and not require_ids:
        print(f"[INFO] {len(meta)} images (patient IDs not needed for enhancement evaluation)")
        return meta
    if missing:
        sys.exit(f"[STOP] {missing} of {len(meta)} images have no patient_id.\n"
                 f"       Open {METADATA_CSV}, fill the patient_id column (and slide_id if\n"
                 f"       known), or set PATIENT_ID_REGEX at the top of this file. Then re-run.")
    meta = load_metadata(METADATA_CSV)
    print(f"[INFO] {len(meta)} images, {meta['patient_id'].nunique()} patients")
    print(meta.groupby("label")["patient_id"].nunique().rename("patients per class").to_string())
    small = meta.groupby("label")["patient_id"].nunique().min()
    if small < 15 and CV_FOLDS is None:
        print(f"[WARN] smallest class has only {small} patients; consider CV_FOLDS = 5")
    return meta


def copy_to_local(meta):
    """Copy images from Drive to the Colab disk once per session (10-50x faster I/O)."""
    if not LOCAL_COPY or not DATASET_DIR or DATASET == "ndbufes":   # ndbufes already local
        return meta
    if not os.path.isdir(LOCAL_COPY):
        print("[INFO] copying images to local disk (one time per session)...")
        t0 = time.time()
        shutil.copytree(DATASET_DIR, LOCAL_COPY)
        print(f"[INFO] copied in {time.time() - t0:.0f}s")
    meta = meta.copy()
    meta["filepath"] = meta["filepath"].str.replace(DATASET_DIR, LOCAL_COPY, regex=False)
    return meta


def base_cfg():
    return dict(img_size=IMG_SIZE, batch=BATCH, epochs=2 if QUICK_TEST else EPOCHS,
                patience=PATIENCE, lr=None, blocks=CSCNN_BLOCKS, stretch=True,
                batchnorm=CSCNN_BATCHNORM, dropout=CSCNN_DROPOUT, stretch_baselines=False)


def run_models(meta, keys, cfg, max_folds=None):
    seeds = [SEEDS[0]] if (QUICK_TEST or CV_FOLDS) else SEEDS
    for key in keys:
        tag = model_tag(key, cfg)
        for seed in seeds:
            for fold, tr, va, te in iter_splits(meta, seed, CV_FOLDS):
                if max_folds and int(fold.replace("fold", "")) >= max_folds:
                    continue
                out = os.path.join(RESULTS_DIR, tag, f"seed{seed}_{fold}")
                if os.path.exists(os.path.join(out, "metrics.json")):
                    print(f"[SKIP] {tag} seed{seed} {fold} already done")
                    continue
                print(f"\n========== {tag} | seed {seed} | {fold} ==========")
                run_one(key, cfg, fold, tr, va, te, seed, out)


def make_all_figures(meta):
    global PROPOSED
    PROPOSED = model_tag("cscnn", base_cfg())
    os.makedirs(FIGURES_DIR, exist_ok=True)
    res = load_results(RESULTS_DIR)
    if not res:
        print("[WARN] no results yet; skipping figures"); return
    print("[INFO] models found:", ", ".join(res))
    plot_stretch_examples(meta, FIGURES_DIR)
    rocs, cms = {}, {}
    for tag, runs in res.items():
        cms[tag] = plot_confusion(tag, runs, FIGURES_DIR)
        rocs[tag] = plot_roc(tag, runs, FIGURES_DIR)
        plot_curves(tag, runs, FIGURES_DIR)
    df, pc = comparison_table(res)
    cols = ["model", "params", "runs", "accuracy", "balanced_accuracy", "balanced_accuracy_ci95",
            "macro_f1", "macro_auc", "ms_per_image"]
    df[cols].to_csv(os.path.join(FIGURES_DIR, "Table_model_comparison.csv"), index=False)
    pc.to_csv(os.path.join(FIGURES_DIR, "Table_per_class.csv"), index=False)
    print(df[cols].to_string(index=False))
    main = {t for t in res if not t.startswith("cscnn") or t == PROPOSED}
    # model comparison on the fold that every model has (baselines run on fold 1 only)
    shared = {t: [r for r in res[t] if r["run"].endswith("fold0")] for t in main}
    shared = {t: v for t, v in shared.items() if v}
    df_s, _ = comparison_table(shared)
    df_s[cols].to_csv(os.path.join(FIGURES_DIR, "Table_model_comparison_fold1_all_models.csv"), index=False)
    print("\n[Comparison on fold 1, all models]\n", df_s[cols].to_string(index=False))
    rocs_s = {t: plot_roc(t, v, os.path.join(FIGURES_DIR, "fold1_only")) for t, v in shared.items()}
    cms_s = {t: plot_confusion(t, v, os.path.join(FIGURES_DIR, "fold1_only")) for t, v in shared.items()}
    plot_comparison(df_s, FIGURES_DIR)
    plot_roc_all(rocs_s, FIGURES_DIR)
    plot_confusion_grid(cms_s, FIGURES_DIR)
    plot_ablation(df, FIGURES_DIR)
    per_magnification_table(res, FIGURES_DIR)
    if PROPOSED in res:
        mcnemar_table(res, FIGURES_DIR)
        try:
            plot_gradcam(PROPOSED, res[PROPOSED], FIGURES_DIR)
            plot_tsne(PROPOSED, res[PROPOSED], FIGURES_DIR)
        except Exception as e:
            print("[WARN] Grad-CAM / t-SNE skipped:", e)
    print(f"[DONE] figures and tables in {FIGURES_DIR}")


def set_dataset(name):
    """Point every path setting at one dataset."""
    g = globals()
    g["DATASET"] = name
    g["DATASET_DIR"] = DATASET_DIRS[name]
    g["WORK_DIR"] = f"{WORK_ROOT}/{name}"
    g["METADATA_CSV"] = f"{g['WORK_DIR']}/metadata.csv"
    g["RESULTS_DIR"] = f"{g['WORK_DIR']}/results_v2" + ("_quicktest" if QUICK_TEST else "")
    g["FIGURES_DIR"] = f"{g['WORK_DIR']}/figures_v2" + ("_quicktest" if QUICK_TEST else "")
    g["LOCAL_COPY"] = f"/content/data_{name}"


def setup_gpu():
    if tf is None:
        sys.exit("[ERROR] TensorFlow is not available.")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[INFO] TensorFlow {tf.__version__}; GPU: {gpus if gpus else 'NONE (very slow!)'}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    if MIXED_PRECISION and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] mixed precision enabled (float16 compute, float32 outputs)")


def run_dataset(name):
    set_dataset(name)
    print(f"\n################ DATASET: {name}  ({DATASET_DIR}) ################")
    if not os.path.isdir(DATASET_DIR):
        print(f"[SKIP] folder not found: {DATASET_DIR}"); return
    meta = prepare_ndbufes() if DATASET == "ndbufes" else prepare_metadata(require_ids=True)
    meta = copy_to_local(meta)
    if "enhancement" in STEPS:
        run_enhancement(meta)
    train_meta = build_cache(meta)
    if "leakage_demo" in STEPS and name in LEAKAGE_DEMO_DATASETS:
        run_leakage_demo(train_meta)
    cfg = base_cfg()
    if "cscnn" in STEPS:
        run_models(train_meta, ["cscnn"], cfg, max_folds=CSCNN_FOLDS)
    if "ablations" in STEPS and not QUICK_TEST:
        run_models(train_meta, ["cscnn"], {**cfg, "stretch": False}, max_folds=CSCNN_FOLDS)   # no stretch
        run_models(train_meta, ["cscnn"], {**cfg, "blocks": 4}, max_folds=CSCNN_FOLDS)        # 4 blocks
        run_models(train_meta, ["cscnn"], {**cfg, "batchnorm": False}, max_folds=CSCNN_FOLDS) # no BN
    if "baselines" in STEPS:
        keys = ["mobilenetv2"] if QUICK_TEST else list(BASELINES)
        run_models(train_meta, keys, cfg, max_folds=BASELINE_FOLDS)
    if "figures" in STEPS:
        make_all_figures(meta)


def main():
    mount_drive()
    setup_gpu()
    t0 = time.time()
    for name in DATASETS_TO_RUN:
        run_dataset(name)
    print(f"\n[ALL DONE] total time {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
