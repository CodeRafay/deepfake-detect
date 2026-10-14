"""deepfake-detect FILE... : scan images and videos from the command line."""
import argparse
import json
import sys

from .model import Detector


def main(argv=None):
    ap = argparse.ArgumentParser(prog="deepfake-detect", description="Scan images or videos for face deepfakes.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--head", help="trained head: local .safetensors or Hugging Face repo id")
    ap.add_argument("--frames", type=int, default=16, help="frames sampled per video (default: 16)")
    ap.add_argument("--device", help="cuda / cpu (default: auto)")
    ap.add_argument("--json", action="store_true", help="one JSON object per line")
    a = ap.parse_args(argv)

    det, failed = Detector(a.head, a.device), False
    for f in a.files:
        try:
            r = det.predict(f, a.frames)
        except (ValueError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            failed = True
            continue
        if a.json:
            print(json.dumps({"file": f, **r}))
        else:
            print(f"{r['label'].upper():<4}  score={r['score']:.3f}  {f}" + (f"  ({r['note']})" if "note" in r else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
