from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(script: str) -> None:
    path = ROOT / script
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    print(f"\n==> {script}")
    subprocess.run([sys.executable, str(path)], check=True, cwd=str(ROOT))

def main():
    (ROOT / "data").mkdir(exist_ok=True)

    # 1) Country -> currency mapping (from REST Countries)
    run("src/currency_map.py")

    # 2) FX momentum (free daily FX snapshot API)
    run("src/fx_momentum_exchange.py")
    run("src/merge_fx_exchange_into_world.py")

    # 3) Macro factors (World Bank)
    run("src/inflation_wb.py")
    run("src/merge_inflation_into_world.py")

    run("src/growth_wb.py")
    run("src/merge_growth_into_world.py")

    run("src/policy_wb.py")
    run("src/merge_policy_into_world.py")

    # 4) Risk (WGI political stability percentile rank)
    run("src/risk_wgi.py")

    print("\nDone. Final map dataset:")
    print("  data/world_fx_exchange_infl_growth_policy_risk.gpkg")

if __name__ == "__main__":
    main()
