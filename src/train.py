import argparse, csv, logging, os
import torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from src.data.dataset import load_cfg, get_loaders
from src.models import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def run_epoch(model, loader, loss_fn, device, opt=None):
    training = opt is not None
    model.train(training)
    total, ys, ps = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(y)
            ys += y.tolist(); ps += out.argmax(1).tolist()
    return total / len(ys), f1_score(ys, ps, average="macro"), accuracy_score(ys, ps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mobilenet", choices=["baseline", "mobilenet"])
    p.add_argument("--tag", required=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--freeze", action="store_true", help="freeze pretrained backbone")
    p.add_argument("--config", default="configs/config.yaml")
    a = p.parse_args()

    cfg = load_cfg(a.config)
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, _, names, weights = get_loaders(cfg, a.batch_size)
    model = build_model(a.model, len(names), a.freeze).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    best = -1.0
    for ep in range(a.epochs):
        tl, tf1, _ = run_epoch(model, train_dl, loss_fn, device, opt)
        vl, vf1, vacc = run_epoch(model, val_dl, loss_fn, device)
        sched.step()
        log.info("[%s] ep %d/%d train_loss=%.3f train_f1=%.3f | val_loss=%.3f val_f1=%.3f val_acc=%.3f",
                 a.tag, ep + 1, a.epochs, tl, tf1, vl, vf1, vacc)
        if vf1 > best:
            best, best_acc = vf1, vacc
            torch.save({"model": a.model, "state": model.state_dict(), "classes": names},
                       f"{cfg['ckpt_dir']}/{a.tag}.pt")

    new = not os.path.exists(cfg["results_csv"])
    with open(cfg["results_csv"], "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "model", "lr", "epochs", "frozen", "best_val_macro_f1", "val_acc"])
        w.writerow([a.tag, a.model, a.lr, a.epochs, a.freeze, round(best, 4), round(best_acc, 4)])


if __name__ == "__main__":
    main()
