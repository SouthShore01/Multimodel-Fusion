import argparse
from pathlib import Path
import urllib.request

DATASET_SOURCES = {
    "autoferry": {
        "homepage": "https://autoferry.github.io/sensor_fusion_dataset/",
        "files": [],
        "notes": "Autoferry dataset is published via project page. Download selected sequences manually from the page and place them under data/autoferry/."
    },
    "whut_msfvessel": {"homepage": "", "files": [], "notes": "Provide official WHUT-MSFVessel links when available."},
    "mtdsp": {"homepage": "", "files": [], "notes": "Provide official MTDSP links when available."},
    "mit_marine_perception": {"homepage": "", "files": [], "notes": "Provide official MIT Marine Perception links when available."},
}


def download(dataset, out_dir):
    out = Path(out_dir) / dataset
    out.mkdir(parents=True, exist_ok=True)
    spec = DATASET_SOURCES.get(dataset, {})
    urls = spec.get("files", [])
    if not urls:
        print(f"No direct downloadable file URLs configured for {dataset}.")
        if spec.get("homepage"):
            print(f"Homepage: {spec['homepage']}")
        if spec.get("notes"):
            print(spec["notes"])
        print(f"Please place selected files under: {out}")
        return
    for u in urls:
        fn = out / u.split("/")[-1]
        print(f"Downloading {u} -> {fn}")
        urllib.request.urlretrieve(u, fn)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["autoferry"], choices=list(DATASET_SOURCES.keys()))
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()
    for ds in args.datasets:
        download(ds, args.out_dir)
