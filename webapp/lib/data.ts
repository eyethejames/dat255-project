import { readFile } from "node:fs/promises";
import path from "node:path";

import type { SeriesOption } from "./types";

type ExportedSeriesData = {
    series_id: string;
    source_series_id: string;
    label: string;
    item_id: string;
    store_id: string;
    state_id: string;
    dept_id: string;
    cat_id: string;
    values: number[];
};

type ExportedSeriesPayload = {
    meta: {
        source_file: string;
        subset: {
            store_id: string;
            cat_id: string;
            dept_id: string;
        };
        num_series: number;
        num_days: number;
    };
    series: ExportedSeriesData[];
};

let cachedPayload: ExportedSeriesPayload | null = null;

async function loadSeriesPayload(): Promise<ExportedSeriesPayload> {
    if (cachedPayload) {
        return cachedPayload;
    }

    const filePath = path.join(
        process.cwd(),
        "data",
        "ca_1_foods_1_validation_series.json"
    );
    const fileContent = await readFile(filePath, "utf-8");
    const parsedPayload = JSON.parse(fileContent) as ExportedSeriesPayload;
    cachedPayload = parsedPayload;
    return parsedPayload;
}

export async function listSeries(): Promise<SeriesOption[]> {
    const payload = await loadSeriesPayload();

    return payload.series.map((series) => ({
        series_id: series.series_id,
        label: `Product ${series.item_id.split("_").at(-1) ?? series.item_id}`,
        description: `${series.cat_id} category, ${series.dept_id} department, ${series.state_id} state. Daily demand history with ${series.values.length} days.`,
        item_id: series.item_id,
        store_id: series.store_id,
        state_id: series.state_id,
        dept_id: series.dept_id,
        cat_id: series.cat_id,
        num_days: series.values.length,
    }));
}

export async function getSeriesById(seriesId: string): Promise<ExportedSeriesData | null> {
    const payload = await loadSeriesPayload();
    return payload.series.find((series) => series.series_id === seriesId) ?? null;
}
