export type SeriesOption = {
    series_id: string;
    label: string;
    description: string;
    item_id: string;
    store_id: string;
    state_id: string;
    dept_id: string;
    cat_id: string;
    num_days: number;
};

export type PolicyResult = {
    total_cost: number;
    stockout_rate: number;
    fill_rate: number;
};

export type InferResponse = {
    series_id: string;
    history: number[];
    target: number[];
    forecasts: {
        point_tcn: number[];
        quantiles: {
            q0_1: number[];
            q0_5: number[];
            q0_9: number[];
        };
    };
    policy_results: {
        baseline: PolicyResult;
        point_tcn: PolicyResult;
        quantile_q0_9: PolicyResult;
    };
    recommended_policy: string;
    meta: {
        input_window: number;
        forecast_horizon: number;
        quantiles: number[];
        inference_backend: string;
        data_source: string;
        point_model: {
            checkpoint_path: string;
            best_epoch: number;
            best_val_loss: number;
            trained_now: boolean;
        };
        quantile_model: {
            checkpoint_path: string;
            best_epoch: number;
            best_val_loss: number;
            trained_now: boolean;
        };
    };
};
