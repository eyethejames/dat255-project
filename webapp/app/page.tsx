"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { InferResponse, PolicyResult, SeriesOption } from "@/lib/types";

function NumberPills({
    values,
    idPrefix,
}: {
    values: number[];
    idPrefix: string;
}) {
    return (
        <div className="number-strip">
            {values.map((value, index) => (
                <span key={`${idPrefix}-${index}`} className="pill">
                    {Number(value).toFixed(2)}
                </span>
            ))}
        </div>
    );
}

function NumberRow({ label, values }: { label: string; values: number[] }) {
    return (
        <div className="result-block">
            <div className="result-label">{label}</div>
            <NumberPills values={values} idPrefix={label} />
        </div>
    );
}

function InfoTooltip({
    label,
    children,
}: {
    label: string;
    children: ReactNode;
}) {
    return (
        <span className="tooltip-wrap">
            <button type="button" className="mini-info-button" aria-label={label}>
                ?
            </button>
            <span className="tooltip-popover" role="tooltip">
                {children}
            </span>
        </span>
    );
}

function CollapsibleSection({
    title,
    children,
    defaultOpen = false,
    className = "",
    actions,
}: {
    title: string;
    children: ReactNode;
    defaultOpen?: boolean;
    className?: string;
    actions?: ReactNode;
}) {
    const [isOpen, setIsOpen] = useState(defaultOpen);

    return (
        <div className={`collapsible-section ${className}`.trim()}>
            <div className="collapsible-header">
                <button
                    type="button"
                    className="collapsible-button"
                    onClick={() => setIsOpen((current) => !current)}
                    aria-expanded={isOpen}
                >
                    <span className="section-title">{title}</span>
                    <span
                        className={`chevron${isOpen ? " open" : ""}`}
                        aria-hidden="true"
                    >
                        ↓
                    </span>
                </button>

                {actions ? <div className="collapsible-actions">{actions}</div> : null}
            </div>

            {isOpen ? <div className="collapsible-content">{children}</div> : null}
        </div>
    );
}

function PolicyCard({
    title,
    result,
    highlight = false,
}: {
    title: string;
    result: PolicyResult;
    highlight?: boolean;
}) {
    return (
        <div className={`policy-card${highlight ? " highlight" : ""}`}>
            <div className="policy-title">{title}</div>
            <div className="policy-metric">
                <span>Total cost</span>
                <strong>{result.total_cost.toFixed(2)}</strong>
            </div>
            <div className="policy-metric">
                <span>Stockout rate</span>
                <strong>{result.stockout_rate.toFixed(3)}</strong>
            </div>
            <div className="policy-metric">
                <span>Fill rate</span>
                <strong>{result.fill_rate.toFixed(3)}</strong>
            </div>
        </div>
    );
}

function ForecastTable({ result }: { result: InferResponse }) {
    const rows = [
        {
            label: "Point forecast",
            description: "Single best estimate",
            values: result.forecasts.point_tcn,
        },
        {
            label: "q0.1 low",
            description: "Low-demand scenario",
            values: result.forecasts.quantiles.q0_1,
        },
        {
            label: "q0.5 median",
            description: "Middle scenario",
            values: result.forecasts.quantiles.q0_5,
        },
        {
            label: "q0.9 high",
            description: "Conservative stockout-aware scenario",
            values: result.forecasts.quantiles.q0_9,
        },
    ];

    return (
        <div className="forecast-table-wrap">
            <table className="forecast-table">
                <thead>
                    <tr>
                        <th scope="col">Forecast type</th>
                        {result.forecasts.point_tcn.map((_, index) => (
                            <th key={`forecast-day-${index}`} scope="col">
                                Day +{index + 1}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row) => (
                        <tr key={row.label}>
                            <th scope="row">
                                <span>{row.label}</span>
                                <small>{row.description}</small>
                            </th>
                            {row.values.map((value, index) => (
                                <td key={`${row.label}-${index}`}>
                                    {Number(value).toFixed(2)}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

function ModelCard({
    title,
    bestEpoch,
    bestValLoss,
    trainedNow,
}: {
    title: string;
    bestEpoch: number;
    bestValLoss: number;
    trainedNow: boolean;
}) {
    return (
        <div className="model-card">
            <div className="policy-title">{title}</div>
            <div className="policy-metric">
                <span>Best epoch</span>
                <strong>{bestEpoch}</strong>
            </div>
            <div className="policy-metric">
                <span>Best validation loss</span>
                <strong>{bestValLoss.toFixed(4)}</strong>
            </div>
            <div className="policy-metric">
                <span>Status</span>
                <strong>{trainedNow ? "Trained now" : "Loaded checkpoint"}</strong>
            </div>
        </div>
    );
}

function buildLinePoints(
    values: number[],
    startIndex: number,
    xForIndex: (index: number) => number,
    yForValue: (value: number) => number
) {
    return values
        .map((value, index) => {
            const prefix = index === 0 ? "M" : "L";
            return `${prefix} ${xForIndex(startIndex + index)} ${yForValue(value)}`;
        })
        .join(" ");
}

function buildBandPath(
    lower: number[],
    upper: number[],
    startIndex: number,
    xForIndex: (index: number) => number,
    yForValue: (value: number) => number
) {
    if (lower.length === 0 || upper.length === 0) {
        return "";
    }

    const upperPath = upper
        .map((value, index) => {
            const prefix = index === 0 ? "M" : "L";
            return `${prefix} ${xForIndex(startIndex + index)} ${yForValue(value)}`;
        })
        .join(" ");

    const lowerPath = lower
        .map((_, reverseIndex) => lower.length - 1 - reverseIndex)
        .map((index) => `L ${xForIndex(startIndex + index)} ${yForValue(lower[index])}`)
        .join(" ");

    return `${upperPath} ${lowerPath} Z`;
}

function ForecastChart({ result }: { result: InferResponse }) {
    const width = 980;
    const height = 380;
    const marginLeft = 78;
    const marginRight = 26;
    const marginTop = 36;
    const marginBottom = 72;
    const chartWidth = width - marginLeft - marginRight;
    const chartHeight = height - marginTop - marginBottom;

    const history = result.history;
    const target = result.target;
    const pointForecast = result.forecasts.point_tcn;
    const q01 = result.forecasts.quantiles.q0_1;
    const q05 = result.forecasts.quantiles.q0_5;
    const q09 = result.forecasts.quantiles.q0_9;

    const totalPoints = history.length + target.length;
    const allValues = [...history, ...target, ...pointForecast, ...q01, ...q05, ...q09];
    const rawMin = Math.min(0, ...allValues);
    const rawMax = Math.max(1, ...allValues);
    const padding = Math.max(0.5, (rawMax - rawMin) * 0.12);
    const minY = rawMin - padding * 0.2;
    const maxY = rawMax + padding;

    const xForIndex = (index: number) =>
        marginLeft + (index / Math.max(totalPoints - 1, 1)) * chartWidth;
    const yForValue = (value: number) =>
        marginTop + ((maxY - value) / Math.max(maxY - minY, 1e-6)) * chartHeight;

    const dividerStep = totalPoints > 1 ? chartWidth / (totalPoints - 1) : 0;
    const dividerX = xForIndex(history.length - 1) + dividerStep / 2;

    const historyPath = buildLinePoints(history, 0, xForIndex, yForValue);
    const targetPath = buildLinePoints(target, history.length, xForIndex, yForValue);
    const pointPath = buildLinePoints(pointForecast, history.length, xForIndex, yForValue);
    const medianPath = buildLinePoints(q05, history.length, xForIndex, yForValue);
    const bandPath = buildBandPath(q01, q09, history.length, xForIndex, yForValue);

    const tickCount = 5;
    const ticks = Array.from({ length: tickCount }, (_, index) => {
        const value = minY + ((maxY - minY) * index) / (tickCount - 1);
        return {
            value,
            y: yForValue(value),
        };
    });

    return (
        <div className="forecast-chart-shell">
            <div className="chart-legend">
                <span className="legend-item">
                    <span className="legend-swatch history" />
                    History window
                </span>
                <span className="legend-item">
                    <span className="legend-swatch target" />
                    Actual target
                </span>
                <span className="legend-item">
                    <span className="legend-swatch point" />
                    Point forecast
                </span>
                <span className="legend-item">
                    <span className="legend-swatch median" />
                    Quantile median
                </span>
                <span className="legend-item">
                    <span className="legend-swatch band" />
                    Quantile band (q0.1-q0.9)
                </span>
            </div>

            <svg
                viewBox={`0 0 ${width} ${height}`}
                className="forecast-chart"
                role="img"
                aria-label="Forecast visualization for selected time series"
            >
                <rect x="0" y="0" width={width} height={height} fill="transparent" />

                {ticks.map((tick) => (
                    <g key={`tick-${tick.value}`}>
                        <line
                            x1={marginLeft}
                            y1={tick.y}
                            x2={width - marginRight}
                            y2={tick.y}
                            stroke="rgba(31, 41, 55, 0.10)"
                            strokeWidth="1"
                        />
                        <text
                            x={marginLeft - 10}
                            y={tick.y + 4}
                            textAnchor="end"
                            fontSize="11"
                            fill="#6b7280"
                        >
                            {tick.value.toFixed(1)}
                        </text>
                    </g>
                ))}

                <line
                    x1={marginLeft}
                    y1={marginTop + chartHeight}
                    x2={width - marginRight}
                    y2={marginTop + chartHeight}
                    stroke="#374151"
                    strokeWidth="1.5"
                />

                <line
                    x1={dividerX}
                    y1={marginTop}
                    x2={dividerX}
                    y2={marginTop + chartHeight}
                    stroke="rgba(234, 122, 41, 0.55)"
                    strokeDasharray="5 5"
                    strokeWidth="1.5"
                />

                <text
                    x={dividerX + 8}
                    y={marginTop + 14}
                    fontSize="11"
                    fontWeight="700"
                    fill="#9a4d14"
                >
                    Forecast starts here
                </text>

                <text x={marginLeft} y={height - 36} fontSize="11" fill="#6b7280">
                    History days
                </text>
                <text x={dividerX + 8} y={height - 36} fontSize="11" fill="#6b7280">
                    Next 7 days
                </text>
                <text
                    x={marginLeft + chartWidth / 2}
                    y={height - 14}
                    textAnchor="middle"
                    fontSize="12"
                    fontWeight="700"
                    fill="#374151"
                >
                    Days from historical input to forecast horizon
                </text>
                <text
                    x={18}
                    y={marginTop + chartHeight / 2}
                    textAnchor="middle"
                    transform={`rotate(-90 18 ${marginTop + chartHeight / 2})`}
                    fontSize="12"
                    fontWeight="700"
                    fill="#374151"
                >
                    Demand units per day
                </text>

                <path d={bandPath} fill="rgba(234, 122, 41, 0.16)" stroke="none" />
                <path
                    d={historyPath}
                    fill="none"
                    stroke="#6b7280"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                />
                <path
                    d={targetPath}
                    fill="none"
                    stroke="#111827"
                    strokeWidth="2.5"
                    strokeDasharray="6 5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                />
                <path
                    d={pointPath}
                    fill="none"
                    stroke="#1d4ed8"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                />
                <path
                    d={medianPath}
                    fill="none"
                    stroke="#ea7a29"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                />
            </svg>
        </div>
    );
}

export default function HomePage() {
    const [series, setSeries] = useState<SeriesOption[]>([]);
    const [selectedSeriesId, setSelectedSeriesId] = useState("");
    const [seriesQuery, setSeriesQuery] = useState("");
    const [result, setResult] = useState<InferResponse | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [isBootstrapping, setIsBootstrapping] = useState(true);
    const [isLoadingResult, setIsLoadingResult] = useState(false);
    const [loadingProgress, setLoadingProgress] = useState(0);
    const resultsRef = useRef<HTMLElement | null>(null);
    const shouldScrollToResultsRef = useRef(false);

    const isLoading = isBootstrapping || isLoadingResult;

    function syncSeriesIdInUrl(seriesId: string) {
        if (typeof window === "undefined") {
            return;
        }

        const nextUrl = new URL(window.location.href);
        if (seriesId) {
            nextUrl.searchParams.set("series_id", seriesId);
        } else {
            nextUrl.searchParams.delete("series_id");
        }
        window.history.replaceState({}, "", nextUrl.toString());
    }

    useEffect(() => {
        async function bootstrap() {
            try {
                const seriesResponse = await fetch("/api/series", { cache: "no-store" });
                if (!seriesResponse.ok) {
                    throw new Error(`Failed to load series list (${seriesResponse.status})`);
                }

                const seriesOptions: SeriesOption[] = await seriesResponse.json();
                setSeries(seriesOptions);

                const requestedSeriesId =
                    typeof window !== "undefined"
                        ? new URLSearchParams(window.location.search).get("series_id")
                        : null;

                const matchedSeriesId = requestedSeriesId
                    ? seriesOptions.find(
                        (option) => option.series_id === requestedSeriesId
                    )?.series_id
                    : null;

                const initialSeriesId = matchedSeriesId ?? "";

                setSelectedSeriesId(initialSeriesId);
                syncSeriesIdInUrl(initialSeriesId);
                setLoadError(null);
            } catch (error) {
                setLoadError(
                    error instanceof Error ? error.message : "Unknown bootstrap error"
                );
            } finally {
                setIsBootstrapping(false);
            }
        }

        void bootstrap();
    }, []);

    useEffect(() => {
        if (isLoading) {
            setLoadingProgress((current) => (current > 0 ? current : 8));

            const intervalId = window.setInterval(() => {
                setLoadingProgress((current) => {
                    if (current >= 94) {
                        return current;
                    }

                    const remaining = 94 - current;
                    const step = Math.max(1.4, remaining * 0.16);
                    return Math.min(94, current + step);
                });
            }, 160);

            return () => window.clearInterval(intervalId);
        }

        if (loadingProgress === 0) {
            return;
        }

        setLoadingProgress(100);
        const timeoutId = window.setTimeout(() => {
            setLoadingProgress(0);
        }, 420);

        return () => window.clearTimeout(timeoutId);
    }, [isLoading]);

    const filteredSeries = useMemo(() => {
        const normalizedQuery = seriesQuery.trim().toLowerCase();
        if (!normalizedQuery) {
            return series;
        }

        return series.filter((option) => {
            return (
                option.series_id.toLowerCase().includes(normalizedQuery) ||
                option.label.toLowerCase().includes(normalizedQuery)
            );
        });
    }, [series, seriesQuery]);

    useEffect(() => {
        if (!result || isLoadingResult || !shouldScrollToResultsRef.current) {
            return;
        }

        shouldScrollToResultsRef.current = false;

        window.setTimeout(() => {
            resultsRef.current?.scrollIntoView({
                behavior: "smooth",
                block: "start",
            });
        }, 90);
    }, [result, isLoadingResult]);

    function handleSeriesSelection(seriesId: string) {
        setSelectedSeriesId(seriesId);
        setResult(null);
        setLoadError(null);
        syncSeriesIdInUrl(seriesId);
    }

    async function handleRunInference(seriesId: string) {
        setIsLoadingResult(true);
        shouldScrollToResultsRef.current = true;
        syncSeriesIdInUrl(seriesId);

        try {
            const response = await fetch(
                `/api/infer?series_id=${encodeURIComponent(seriesId)}`,
                { cache: "no-store" }
            );
            if (!response.ok) {
                throw new Error(`Inference request failed (${response.status})`);
            }

            const payload: InferResponse = await response.json();
            setResult(payload);
            setLoadError(null);
        } catch (error) {
            setLoadError(
                error instanceof Error ? error.message : "Unknown inference error"
            );
        } finally {
            setIsLoadingResult(false);
        }
    }

    return (
        <main className="page-shell">
            <section className="hero-card">
                <div className="hero-meta">
                    <div className="eyebrow">DAT255 project demo</div>
                    <div className="hero-links">
                        <a href="/explain" className="repo-link">
                            Plain-language guide
                        </a>
                        <a
                            href="https://github.com/eyethejames/dat255-project"
                            className="repo-link"
                            target="_blank"
                            rel="noreferrer"
                        >
                            GitHub repository
                        </a>
                    </div>
                </div>
                <h1>An Uncertainty-Aware Demand Forecasting Model for Inventory Restocking</h1>
                <p className="hero-copy">
                    This interactive case study shows how a deep learning forecasting model
                    can support inventory planning. Select one product series, run inference,
                    inspect the forecast horizon, and compare how different restocking
                    policies would have performed against the actual outcome.
                </p>

                <div className="intro-grid">
                    <article className="intro-card">
                        <CollapsibleSection title="What the demo does">
                            <p className="section-subtitle">
                                The app lets the user select a product time series, run a
                                trained forecasting model, inspect point and quantile forecasts,
                                and compare inventory-oriented policy outcomes such as total
                                cost, stockout rate and fill rate.
                            </p>
                        </CollapsibleSection>
                    </article>

                    <article className="intro-card">
                        <CollapsibleSection title="How to use it">
                            <div className="step-list">
                                <div className="step-item">
                                    <span className="step-index">1</span>
                                    <span>Search for a product series in the subset.</span>
                                </div>
                                <div className="step-item">
                                    <span className="step-index">2</span>
                                    <span>
                                        Click <strong>Run inference</strong> to generate a new
                                        forecast.
                                    </span>
                                </div>
                                <div className="step-item">
                                    <span className="step-index">3</span>
                                    <span>
                                        Compare the forecast output and the recommended policy.
                                    </span>
                                </div>
                            </div>
                        </CollapsibleSection>
                    </article>
                </div>
            </section>

            <section className="controls-card">
                <div>
                    <div className="section-title">Series selection</div>
                    <div className="section-subtitle">
                        Browse real exported M5 series from the CA_1 / FOODS / FOODS_1 subset
                        and run real model inference through the API layer.
                    </div>
                </div>

                <div className="controls-row">
                    <input
                        value={seriesQuery}
                        onChange={(event) => setSeriesQuery(event.target.value)}
                        placeholder="Search by item or series id"
                        className="series-select"
                        disabled={isBootstrapping}
                    />
                    <select
                        value={selectedSeriesId}
                        onChange={(event) => handleSeriesSelection(event.target.value)}
                        className="series-select"
                        disabled={isBootstrapping}
                    >
                        <option value="">Select a series</option>
                        {filteredSeries.map((option) => (
                            <option key={option.series_id} value={option.series_id}>
                                {option.label}
                            </option>
                        ))}
                    </select>

                    <button
                        className="primary-button"
                        onClick={() => void handleRunInference(selectedSeriesId)}
                        disabled={isBootstrapping || isLoadingResult || !selectedSeriesId}
                    >
                        {isLoadingResult ? "Running..." : "Run inference"}
                    </button>
                </div>

                <div className={`loading-track${loadingProgress > 0 ? " visible" : ""}`}>
                    <div
                        className="loading-bar"
                        style={{ width: `${loadingProgress}%` }}
                    />
                </div>

                {loadError ? <div className="error-banner">{loadError}</div> : null}
            </section>

            <section ref={resultsRef} className="results-grid">
                <article className="result-card">
                    <div className="title-row">
                        <div className="section-title">Demand data</div>
                        <InfoTooltip label="Explain demand data">
                            Daily demand means the number of units sold or requested for this
                            product series on one day. A value of 3.00 means demand was 3 units
                            that day.
                        </InfoTooltip>
                    </div>
                    {isBootstrapping ? (
                        <p className="muted">Loading real exported series...</p>
                    ) : (
                        <>
                            <div className="stat-row">
                                <span>Available series</span>
                                <strong>{series.length}</strong>
                            </div>
                            <div className="stat-row">
                                <span>Visible in current search</span>
                                <strong>{filteredSeries.length}</strong>
                            </div>
                            <div className="stat-row">
                                <span>Selected series</span>
                                <strong>{selectedSeriesId || "-"}</strong>
                            </div>
                            {result ? (
                                <>
                                    <div className="stat-row">
                                        <span>Displayed result for</span>
                                        <strong>{result.series_id}</strong>
                                    </div>
                                    <div className="stat-row">
                                        <span>Inference backend</span>
                                        <strong>{result.meta.inference_backend}</strong>
                                    </div>
                                    <div className="stat-row">
                                        <span>Data source</span>
                                        <strong>{result.meta.data_source}</strong>
                                    </div>
                                </>
                            ) : null}
                            {result ? (
                                <>
                                    <p className="context-note">
                                        The model receives the last 28 observed daily demand
                                        values as input. The next 7 days are the actual outcome
                                        used to compare the forecast against reality.
                                    </p>
                                    <CollapsibleSection
                                        title="Observed demand input (last 28 days)"
                                        className="result-block"
                                        actions={
                                            <InfoTooltip label="Explain observed demand input">
                                                Each box is one historical day. The number is
                                                how many units were demanded or sold that day.
                                                These 28 values are the model input.
                                            </InfoTooltip>
                                        }
                                    >
                                        <NumberPills
                                            values={result.history}
                                            idPrefix="Observed demand input (last 28 days)"
                                        />
                                    </CollapsibleSection>
                                    <NumberRow
                                        label="Actual demand outcome (next 7 days)"
                                        values={result.target}
                                    />
                                </>
                            ) : null}
                        </>
                    )}
                </article>

                <article className="result-card">
                    <div className="title-row">
                        <div className="section-title">Model forecasts for the next 7 days</div>
                        <InfoTooltip label="Explain forecast table">
                            Each column is one future day. Point forecast is one best estimate.
                            The quantile rows show low, median, and high demand scenarios.
                        </InfoTooltip>
                    </div>
                    <p className="context-note">
                        Forecast values are predicted demand units. For example, 2.00 means
                        the model expects about 2 units of demand on that future day.
                    </p>
                    {result ? (
                        <CollapsibleSection
                            title="Forecast table"
                            className="result-block"
                            defaultOpen
                        >
                            <ForecastTable result={result} />
                        </CollapsibleSection>
                    ) : (
                        <p className="muted">Inference result will appear here.</p>
                    )}
                </article>
            </section>

            <section className="policies-card">
                <div className="title-row">
                    <div className="section-title">Forecast visualization</div>
                    <InfoTooltip label="Explain forecast visualization">
                        The grey line is what the model saw. The dashed black line is what
                        actually happened. Blue and orange lines are model forecasts.
                    </InfoTooltip>
                </div>
                <div className="section-subtitle">
                    The chart shows historical demand, actual future demand, the point forecast,
                    and the uncertainty band between q0.1 and q0.9.
                </div>

                {result ? (
                    <ForecastChart result={result} />
                ) : (
                    <p className="muted">Run inference to visualize the forecast horizon.</p>
                )}
            </section>

            <section className="policies-card">
                <div className="title-row">
                    <div className="section-title">Policy recommendation</div>
                    <InfoTooltip label="Explain policy recommendation">
                        A policy turns a forecast into a restocking decision. The recommended
                        policy is the one with lowest simulated total cost for this selected
                        series and 7-day outcome.
                    </InfoTooltip>
                </div>
                <div className="section-subtitle">
                    Decision metrics compare how different restocking rules would have performed
                    against the actual next 7 days.
                </div>

                {result ? (
                    <>
                        <div className="recommendation-banner">
                            Recommended policy: <strong>{result.recommended_policy}</strong>
                            <span className="recommendation-note">
                                {" "}
                                · selected by lowest simulated total cost for this series
                            </span>
                        </div>

                        <div className="policy-grid">
                            <PolicyCard
                                title="Baseline"
                                result={result.policy_results.baseline}
                            />
                            <PolicyCard
                                title="Point TCN"
                                result={result.policy_results.point_tcn}
                            />
                            <PolicyCard
                                title="Quantile q0.9"
                                result={result.policy_results.quantile_q0_9}
                                highlight
                            />
                        </div>

                        <div className="metrics-note">
                            <div className="metrics-note-title">
                                How to read the policy metrics
                            </div>
                            <p>
                                <strong>Total cost</strong> is a simulated penalty score:
                                ordering too much adds holding cost, while ordering too little
                                adds stockout cost. This demo uses holding cost = 1 and
                                stockout cost = 5, so missing demand is penalized more heavily
                                than carrying extra inventory. Lower is better.
                            </p>
                            <p>
                                <strong>Stockout rate</strong> is the share of forecast days
                                where demand was higher than available inventory. Lower is better.
                            </p>
                            <p>
                                <strong>Fill rate</strong> is the share of actual demand the
                                policy was able to satisfy. Higher is better.
                            </p>
                        </div>
                    </>
                ) : (
                    <p className="muted">Run inference to see policy metrics.</p>
                )}
            </section>

            <section className="policies-card">
                <div className="section-title">Model diagnostics</div>
                <div className="section-subtitle">
                    The API returns checkpoint metadata from the real point and quantile TCN
                    models used for inference.
                </div>

                {result ? (
                    <div className="policy-grid">
                        <ModelCard
                            title="Point TCN checkpoint"
                            bestEpoch={result.meta.point_model.best_epoch}
                            bestValLoss={result.meta.point_model.best_val_loss}
                            trainedNow={result.meta.point_model.trained_now}
                        />
                        <ModelCard
                            title="Quantile TCN checkpoint"
                            bestEpoch={result.meta.quantile_model.best_epoch}
                            bestValLoss={result.meta.quantile_model.best_val_loss}
                            trainedNow={result.meta.quantile_model.trained_now}
                        />
                    </div>
                ) : (
                    <p className="muted">Run inference to inspect checkpoint metadata.</p>
                )}
            </section>
        </main>
    );
}
