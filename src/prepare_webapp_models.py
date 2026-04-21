import json

from real_tcn_inference import warm_models


def main():
    payload = warm_models()
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
