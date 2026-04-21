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
        label: `${series.item_id} · ${series.series_id}`,
    }));
}

export async function getSeriesById(seriesId: string): Promise<ExportedSeriesData | null> {
    const payload = await loadSeriesPayload();
    return payload.series.find((series) => series.series_id === seriesId) ?? null;
}
