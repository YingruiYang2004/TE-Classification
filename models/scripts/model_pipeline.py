import os, torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm

from scripts.seq_dataset import collate_pad, prepare_data
from scripts.cnn_module import RCInputInvariantCNN

def resolve_device(requested=None):
    """Return the best available accelerator as a torch.device."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def save_torch(data, path, filename): 
    os.makedirs(path, exist_ok=True)
    if os.path.exists("%s/%s.pt"%(path, filename)):
        n = 1
        while n <= 100:
            if os.path.exists("%s/%s_legend_%s.pt"%(path, filename, n)):
                n += 1
            else:
                torch.save(data, "%s/%s_legend_%s.pt"%(path, filename, n))
                break
    else:
        torch.save(data, "%s/%s.pt"%(path, filename))
        
def train_one_epoch(model, loader, opt, loss_fn, device, ep, epochs):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {ep}/{epochs}", leave=False)
    for _, X, mask, Y in pbar:
        X, mask, Y = X.to(device), mask.to(device), Y.to(device)
        opt.zero_grad()
        logits = model(X, mask)
        loss = loss_fn(logits, Y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * X.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def evaluate(model, loader, loss_fn, device, dataset):
    model.eval()
    val_running = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for _, X, mask, Y in tqdm(loader, desc="Val", leave=False):
            X, mask = X.to(device, dtype=torch.float32), mask.to(device)
            Y, logits = Y.to(device, dtype=torch.float32), model(X, mask)
            loss = loss_fn(logits, Y)
            val_running += loss.item() * X.size(0)
            all_logits.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(Y.detach().cpu().numpy())
    all_logits = np.concatenate(all_logits) if all_logits else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    val_loss = val_running / len(dataset) if len(dataset) else float("nan")
    
    try:
        auroc = roc_auc_score(all_labels, all_logits)
        auprc = average_precision_score(all_labels, all_logits)
    except ValueError:
        auroc, auprc = float("nan"), float("nan")
    return auroc, auprc, val_loss, all_logits, all_labels

def run_train(fasta_path, label_path, batch_size=8, num_workers=0, epochs=5, lr=1e-3, device=None, patience=20, subset_size=None, random_state=42, trial=False):
    
    device = resolve_device(device)
    print(f"Using device: {device}")

    ds_tr, ds_val, ds_te = prepare_data(fasta_path, label_path, subset_size=subset_size, random_state=random_state)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_pad)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_pad)
    loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_pad)

    model = RCInputInvariantCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Starting training...")

    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": []}
    best_state, best_epoch = None, None
    best_metrics = {"auroc": -float("inf"), "auprc": float("nan"), "val_loss": float("inf")}
    best_scores, best_labels = None, None
    last_scores, last_labels = None, None
    last_val_loss = None
    last_auroc, last_auprc = None, None
    bad = 0
    patience = patience if patience is not None else epochs + 1

    for ep in range(1, epochs + 1):
        
        train_loss = train_one_epoch(model, loader_tr, opt, loss_fn, device, ep, epochs)
        
        auroc, auprc, val_loss, Ps, Ys = evaluate(model, loader_val, loss_fn, device, ds_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(auroc)
        history["val_auprc"].append(auprc)

        print(f"Epoch {ep}: train loss {train_loss:.4f} | Val loss {val_loss:.4f} | Val AUROC {auroc:.4f} | AUPRC {auprc:.4f}")

        last_scores, last_labels = Ps, Ys
        last_val_loss, last_auroc, last_auprc = val_loss, auroc, auprc

        improved = auroc > best_metrics["auroc"] + 1e-4
        if improved:
            best_metrics = {"auroc": auroc, "auprc": auprc, "val_loss": val_loss}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_scores, best_labels, best_epoch = Ps, Ys, ep
            bad = 0
            torch.save(best_state, "vgp_model_data_tpase/rc_cnn_latest.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break
        if auroc == 1:
            print("Early stopping.")
            break
    
    if best_state is not None:
        if not trial:
            save_torch(best_state, "vgp_model_data_tpase", "rc_cnn_best")
            model.load_state_dict(best_state)
            model.to(device)
    else:
        best_scores, best_labels = last_scores, last_labels
        best_epoch = best_epoch if best_epoch is not None else len(history["train_loss"])
        best_metrics = {
            "auroc": last_auroc,
            "auprc": last_auprc,
            "val_loss": last_val_loss
        }
    
    auroc, auprc, te_loss, te_scores, te_labels = evaluate(model, loader_te, loss_fn, device, ds_te)
    print(f"Test set performance: Loss {te_loss:.4f} | AUROC {auroc:.4f} | AUPRC {auprc:.4f}")

    results = {
        "model": model,
        "history": history,
        "metrics": {
            "best_epoch": best_epoch,
            "best_auroc": best_metrics["auroc"],
            "best_auprc": best_metrics["auprc"],
            "best_val_loss": best_metrics["val_loss"], 
            "test_auroc": auroc, 
            "test_auprc": auprc, 
            "test_loss": te_loss
        },
        "roc": {
            "labels": te_labels,
            "scores": te_scores, 
            "best_labels": best_labels,
            "best_scores": best_scores
        },
        "device": str(device)
    }
    return results