import { HOLDING_COST, STOCKOUT_COST } from "./constants";
import type { PolicyResult } from "./types";

export function inventorySimulation(
    forecast: number[],
    actual: number[],
    holdingCost = HOLDING_COST,
    stockoutCost = STOCKOUT_COST
): PolicyResult {
    let totalCost = 0;
    let stockoutDays = 0;
    let fulfilledDemand = 0;
    let totalDemand = 0;

    for (let i = 0; i < actual.length; i++) {
        const predicted = Math.max(0, Math.round(forecast[i] ?? 0));
        const demand = actual[i] ?? 0;

        const fulfilled = Math.min(predicted, demand);
        const leftover = Math.max(0, predicted - demand);
        const unmet = Math.max(0, demand - predicted);

        totalCost += leftover * holdingCost + unmet * stockoutCost;

        if (unmet > 0) stockoutDays += 1;

        fulfilledDemand += fulfilled;
        totalDemand += demand;
    }

    return {
        total_cost: totalCost,
        stockout_rate: actual.length > 0 ? stockoutDays / actual.length : 0,
        fill_rate: totalDemand > 0 ? fulfilledDemand / totalDemand : 0,
    };
}