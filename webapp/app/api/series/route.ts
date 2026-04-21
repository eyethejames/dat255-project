import { NextResponse } from "next/server";
import { listSeries } from "@/lib/data";

export const runtime = "nodejs";

export async function GET() {
    const series = await listSeries();
    return NextResponse.json(series);
}
