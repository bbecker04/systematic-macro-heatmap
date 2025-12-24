from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)

    gdf = gpd.read_file("data/world_fx.gpkg", layer="countries_fx")

    fig, ax = plt.subplots(figsize=(16, 8))
    gdf.plot(
        column="fx_score_mom",
        legend=True,
        ax=ax,
    )

    ax.set_axis_off()
    ax.set_title("FX Momentum Score (0–100) — Frankfurter (base EUR)")

    out_path = Path("outputs/fx_momentum_map.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)
