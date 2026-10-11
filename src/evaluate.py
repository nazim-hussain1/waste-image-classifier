import argparse, logging, os
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from src.data.dataset import load_cfg, get_splits, ImgDS, eval_tf
from src.models import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--config", default="configs/config.yaml")
    a = p.parse_args()
    cfg = load_cfg(a.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    ck = torch.load(f"{cfg['ckpt_dir']}/{a.tag}.pt", map_location=device)
    names = ck["classes"]
    model = build_model(ck["model"], len(names)).to(device)
    model.load_state_dict(ck["state"]); model.eval()

    _, _, test_hf, _ = get_splits(cfg)
    dl = DataLoader(ImgDS(test_hf, eval_tf(cfg["img_size"])), batch_size=64)
    ys, ps, conf = [], [], []
    with torch.no_grad():
        for x, y in dl:
            pr = torch.softmax(model(x.to(device)), 1).cpu()
            ys += y.tolist(); ps += pr.argmax(1).tolist(); conf += pr.max(1).values.tolist()
    ys, ps, conf = np.array(ys), np.array(ps), np.array(conf)

    report = classification_report(ys, ps, target_names=names, digits=3)
    print(report)
    with open(f"{cfg['fig_dir']}/{a.tag}_test_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(ys, ps)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red" if i != j and cm[i, j] else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix - {a.tag}")
    plt.tight_layout(); plt.savefig(f"{cfg['fig_dir']}/{a.tag}_confusion_matrix.png", dpi=150); plt.close()

    wrong = np.where(ys != ps)[0]
    wrong = wrong[np.argsort(-conf[wrong])][:16]   # most confidently wrong first
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(test_hf[int(i)]["image"])
        ax.set_title(f"true: {names[ys[i]]}\npred: {names[ps[i]]} ({conf[i]:.2f})", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{cfg['fig_dir']}/{a.tag}_misclassified.png", dpi=120); plt.close()
    log.info("Saved figures to %s (%d errors out of %d)", cfg["fig_dir"], (ys != ps).sum(), len(ys))


if __name__ == "__main__":
    main()
