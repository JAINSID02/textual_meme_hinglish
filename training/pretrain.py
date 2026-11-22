# meme_llm/training/pretrain.py
import argparse
import math
import signal
import sys
import time
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg
from tokenizer.train_tokenizer import HinglishTokenizer
from model.transformer import TransformerLM
from training.lm_dataset import make_loaders
from training.utils import (set_seed, get_device, WarmupCosine, build_optimizer,
                            save_checkpoint, load_checkpoint, prune_checkpoints)


def cycle(loader):
    """Loop over a DataLoader forever (reshuffles each pass)."""
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, val_loader, device, use_amp, max_batches=20):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps (default: PretrainConfig.steps)")
    ap.add_argument("--resume", action="store_true", help="continue from runs/pretrain/model_last.pt")
    args = ap.parse_args()

    pc, mc = cfg.pretrain, cfg.model
    set_seed(cfg.cls.seed)
    device = get_device()
    use_amp = pc.mixed_precision and device.type == "cuda"
    out_dir = Path(cfg.paths.pretrain_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = HinglishTokenizer()
    assert tok.vocab_size == mc.vocab_size, \
        f"tokenizer vocab {tok.vocab_size} != ModelConfig.vocab_size {mc.vocab_size}"
    train_loader, val_loader, steps_per_epoch = make_loaders(tok)
    total_steps = args.steps or pc.steps
    print(f"[pretrain] total optimizer steps = {total_steps} "
          f"(= {total_steps / max(1, steps_per_epoch):.2f} epochs)")

    model = TransformerLM.from_config(mc).to(device)
    print(f"[pretrain] params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M on {device}")
    opt = build_optimizer(model, pc.lr, pc.weight_decay)
    sched = WarmupCosine(opt, pc.lr, pc.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    last_path = out_dir / "model_last.pt"
    if args.resume and last_path.exists():
        ck = load_checkpoint(last_path, model, opt, scaler, map_location=device)
        step = ck["step"]
        print(f"[pretrain] resumed from step {step}")

    # graceful stop: finish the current step, save, exit
    stop = {"flag": False}
    def _on_signal(sig, frame):
        stop["flag"] = True
        print("\n[pretrain] stop requested, will save after this step...")
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    best_val = float("inf")
    batches = cycle(train_loader)
    model.train()
    t0 = time.time()

    while step < total_steps and not stop["flag"]:
        lr = sched.set(step)
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0

        for _ in range(pc.grad_accum_steps):
            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
            scaler.scale(loss / pc.grad_accum_steps).backward()
            loss_acc += loss.item() / pc.grad_accum_steps

        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % 10 == 0 or step == 1:
            dt = time.time() - t0
            print(f"step {step:5d}/{total_steps} | loss {loss_acc:.4f} | lr {lr:.2e} "
                  f"| gnorm {float(grad_norm):.2f} | {dt:.1f}s")
            t0 = time.time()

        if step % pc.save_every == 0 or step == total_steps:
            val = evaluate(model, val_loader, device, use_amp)
            print(f"[eval] step {step} | val loss {val:.4f} | val ppl {math.exp(val):.1f}")
            save_checkpoint(last_path, model, opt, scaler, step)
            save_checkpoint(out_dir / f"step_{step:07d}.pt", model, None, None, step)
            prune_checkpoints(out_dir, "step_*.pt", keep_last=2)
            if val < best_val:
                best_val = val
                save_checkpoint(out_dir / "model_best.pt", model, None, None, step,
                                extra={"val_loss": val})
                print(f"[eval] new best -> model_best.pt")

    # final / interrupted save
    save_checkpoint(last_path, model, opt, scaler, step)
    print(f"[pretrain] saved {last_path} at step {step}")


if __name__ == "__main__":
    main()