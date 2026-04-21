import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";

import { FORECAST_HORIZON, INPUT_WINDOW } from "./constants";
import type { InferResponse } from "./types";

const execFileAsync = promisify(execFile);

export async function runInferenceForSeries(
    seriesId: string,
    values: number[]
): Promise<InferResponse> {
    if (values.length < INPUT_WINDOW + FORECAST_HORIZON) {
        throw new Error("Series is too short for real inference.");
    }

    const projectRoot = path.resolve(process.cwd(), "..");
    const scriptPath = path.join(projectRoot, "src", "webapp_inference_runtime.py");
    const pythonExecutable = process.env.PYTHON_EXECUTABLE || "python3";

    const { stdout, stderr } = await execFileAsync(
        pythonExecutable,
        [scriptPath, "--series-id", seriesId],
        {
            cwd: projectRoot,
            env: process.env,
            maxBuffer: 10 * 1024 * 1024,
        }
    );

    if (stderr.trim()) {
        console.error(stderr);
    }

    const parsed = JSON.parse(stdout) as InferResponse | { error: string };
    if ("error" in parsed) {
        throw new Error(parsed.error);
    }

    return parsed;
}
