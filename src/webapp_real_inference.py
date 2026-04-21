import argparse
import json
import sys

from real_tcn_inference import run_real_inference, warm_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-id", dest="series_id")
    parser.add_argument("--warmup-only", action="store_true")
    args = parser.parse_args()

    try:
        if args.warmup_only:
            payload = warm_models()
        else:
            if not args.series_id:
                raise ValueError("--series-id is required unless --warmup-only is used.")
            payload = run_real_inference(args.series_id)
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"error": str(error)}, ensure_ascii=True), file=sys.stdout)
        sys.exit(1)

    print(json.dumps(payload, ensure_ascii=True), file=sys.stdout)


if __name__ == "__main__":
    main()
