"""Train the PPO proposer for E4 (stable-baselines3). Development data only.

Usage (from the code folder):
    python scripts/train_ppo.py --smoke     # ~1-2 minutes: checks that everything runs
    python scripts/train_ppo.py             # grid (2 x 2 configs x 3 seeds) on the eval period, then 10 final seeds
Training period 2003-2017; hyper-parameters chosen by mean Sharpe on 2018-2021 (development "validation").
Saves models to data/interim/e4/ppo/ and results/e4_ppo/{grid.csv, log.txt, env_info.txt, handoff_e4_ppo.zip}.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from reguard.data.store import load_panel  # noqa: E402
from reguard.portfolio.backtest import run_backtest  # noqa: E402
from reguard.portfolio.data import Market  # noqa: E402
from reguard.portfolio.metrics import summary  # noqa: E402
from reguard.portfolio.rl_env import ObsBuilder, PortfolioEnv, ppo_proposer  # noqa: E402
from reguard.utils.handoff import CODE_ROOT, PROJECT_ROOT, Tee, make_zip, step_dir, write_env_info  # noqa: E402

STEP = "e4_ppo"
MODEL_DIR = PROJECT_ROOT / "data" / "interim" / "e4" / "ppo"


def train(mkt, ob, cfg, lr, ent, seed, steps):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    p = cfg["ppo"]
    a, b = cfg["periods"]["ppo_train"]
    torch.manual_seed(seed)
    env = DummyVecEnv([lambda: PortfolioEnv(mkt, ob, a, b, p["episode_weeks"], cfg["rebalance_every"], cfg["cost_bps"], seed)])
    model = PPO("MlpPolicy", env, learning_rate=lr, ent_coef=ent, n_steps=p["n_steps"], batch_size=p["batch_size"],
                gamma=p["gamma"], seed=seed, verbose=0, device="cpu",
                policy_kwargs={"net_arch": list(p["net_arch"])})
    model.learn(total_timesteps=steps)
    return model


def evaluate(model, mkt, ob, cfg):
    a, b = cfg["periods"]["eval"]
    bt = run_backtest(mkt, a, b, ppo_proposer(model, ob), pd.DataFrame(), cfg, governed=False, pool=None)
    return bt, summary(bt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    d = step_dir(STEP)
    tee = Tee(d / "log.txt"); sys.stdout = tee
    with open(CODE_ROOT / "configs" / "experiment" / "e4.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    write_env_info(d, seed=cfg["seed"])
    p = cfg["ppo"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mkt = Market(load_panel(cfg["panel"]), cfg["min_history"])
    ob = ObsBuilder(mkt, p["obs_features"])
    steps = 2048 if args.smoke else p["total_timesteps"]
    grid = list(itertools.product(p["grid"]["learning_rate"], p["grid"]["ent_coef"]))
    if args.smoke:
        grid = grid[:1]
    seeds = [0] if args.smoke else list(range(p["grid_seeds"]))
    T0 = time.time()
    rows, rets = [], {}
    for lr, ent in grid:
        for s in seeds:
            t0 = time.time()
            m = train(mkt, ob, cfg, lr, ent, 1000 + s, steps)
            bt, sm = evaluate(m, mkt, ob, cfg)
            rets[f"lr={lr}|ent={ent}|seed={s}"] = bt.set_index("date")["net"]
            rows.append({"lr": lr, "ent_coef": ent, "seed": s, **sm, "seconds": round(time.time() - t0, 1)})
            print(f"  grid lr={lr} ent={ent} seed={s}: Sharpe {sm['sharpe']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    G = pd.DataFrame(rows)
    G.to_csv(d / "grid.csv", index=False)
    pd.DataFrame(rets).to_csv(MODEL_DIR / "grid_eval_returns.csv")
    best = G.groupby(["lr", "ent_coef"])["sharpe"].mean().idxmax()
    print(f"  chosen: lr={best[0]} ent_coef={best[1]} (mean eval Sharpe over seeds)", flush=True)
    finals = [0] if args.smoke else list(range(p["final_seeds"]))
    for s in finals:
        t0 = time.time()
        m = train(mkt, ob, cfg, best[0], best[1], s, steps)
        m.save(MODEL_DIR / f"final_seed{s}.zip")
        print(f"  final seed {s} saved ({time.time()-t0:.0f}s)", flush=True)
    (MODEL_DIR / "chosen.yaml").write_text(yaml.safe_dump({"learning_rate": float(best[0]), "ent_coef": float(best[1]),
                                                            "smoke": bool(args.smoke), "seeds": finals}), encoding="utf-8")
    print(f"done in {time.time()-T0:.0f}s", flush=True)
    sys.stdout = tee.stdout; tee.close()
    print(f"Handoff: {make_zip(d, STEP)}")


if __name__ == "__main__":
    main()
