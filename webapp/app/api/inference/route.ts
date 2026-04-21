import { NextRequest, NextResponse } from "next/server";
import { getSeriesById } from "@/lib/data";
import { runInferenceForSeries } from "@/lib/inference";

export const runtime = "nodejs";

export async function GET(request: NextRequest) {
    const { searchParams } = new URL(request.url);
    const seriesId = searchParams.get("series_id");

    if (!seriesId) {
        return NextResponse.json(
            { error: "Missing required query parameter: series_id" },
            { status: 400 }
        );
    }

    const series = await getSeriesById(seriesId);

    if (!series) {
        return NextResponse.json(
            { error: `Series not found: ${seriesId}` },
            { status: 404 }
        );
    }

    try {
        const result = await runInferenceForSeries(series.series_id, series.values);
        return NextResponse.json(result);
    } catch (error) {
        const message =
            error instanceof Error ? error.message : "Unknown inference error";

        return NextResponse.json({ error: message }, { status: 500 });
    }
}
