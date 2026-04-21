import json
from pathlib import Path

import load_data


STORE_ID = "CA_1"
CAT_ID = "FOODS"
DEPT_ID = "FOODS_1"
OUTPUT_PATH = Path("webapp/data/ca_1_foods_1_validation_series.json")


def normalize_series_id(raw_series_id: str) -> str:
    if raw_series_id.endswith("_validation"):
        return raw_series_id[: -len("_validation")]
    return raw_series_id


def build_export_payload() -> dict:
    sales = load_data.sales
    subset = sales[
        (sales["store_id"] == STORE_ID)
        & (sales["cat_id"] == CAT_ID)
        & (sales["dept_id"] == DEPT_ID)
    ].copy()

    day_cols = [column for column in subset.columns if column.startswith("d_")]
    subset = subset.sort_values("id").reset_index(drop=True)

    exported_series = []
    for _, row in subset.iterrows():
        raw_series_id = row["id"]
        exported_series.append(
            {
                "series_id": normalize_series_id(raw_series_id),
                "source_series_id": raw_series_id,
                "label": normalize_series_id(raw_series_id),
                "item_id": row["item_id"],
                "store_id": row["store_id"],
                "state_id": row["state_id"],
                "dept_id": row["dept_id"],
                "cat_id": row["cat_id"],
                "values": [int(row[column]) for column in day_cols],
            }
        )

    return {
        "meta": {
            "source_file": "sales_train_validation.csv",
            "subset": {
                "store_id": STORE_ID,
                "cat_id": CAT_ID,
                "dept_id": DEPT_ID,
            },
            "num_series": len(exported_series),
            "num_days": len(day_cols),
        },
        "series": exported_series,
    }


def main() -> None:
    payload = build_export_payload()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )

    print(f"Exported {payload['meta']['num_series']} series to {OUTPUT_PATH}")
    print(f"Days per series: {payload['meta']['num_days']}")


if __name__ == "__main__":
    main()
