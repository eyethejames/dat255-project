import type { Metadata } from "next";

import DictionaryGuide from "./DictionaryGuide";

export const metadata: Metadata = {
    title: "Plain-Language Guide | Demand Forecast Demo",
    description:
        "Simple explanation of time series forecasting, inventory metrics and the model behind the demo.",
};

const dictionaryEntries = [
    {
        term: "Time series",
        short: "A product's demand history ordered by day.",
        detail:
            "In this demo, one time series means one product in one store. Each number is how many units were demanded on one day.",
    },
    {
        term: "Daily demand",
        short: "How many units customers wanted or bought that day.",
        detail:
            "A value of 3 means the store needed 3 units of that product on that day. Zero means there was no observed demand that day.",
    },
    {
        term: "Forecast",
        short: "The model's estimate of future demand.",
        detail:
            "The forecast is not a guarantee. It is a prediction based on recent demand patterns in the selected product series.",
    },
    {
        term: "Quantile forecast",
        short: "Low, middle and high demand scenarios.",
        detail:
            "Instead of giving only one number, the quantile model gives scenarios. q0.9 is a high-demand scenario and can be useful when avoiding stockouts is important.",
    },
    {
        term: "Restocking policy",
        short: "A rule for how much inventory to prepare.",
        detail:
            "The demo turns forecasts into simple inventory decisions and then compares how those decisions would have performed.",
    },
    {
        term: "Stockout rate",
        short: "How often the policy did not stock enough.",
        detail:
            "Calculated as stockout days divided by total forecast days. Lower is better because it means fewer days where demand could not be fully served.",
    },
    {
        term: "Fill rate",
        short: "How much customer demand was satisfied.",
        detail:
            "Calculated as fulfilled demand divided by total demand. Higher is better because it means more customer demand was actually served.",
    },
    {
        term: "Total cost",
        short: "A simulated penalty for the inventory decision.",
        detail:
            "The demo adds a small penalty for extra inventory and a larger penalty for missed demand. Lower total cost means a better decision under these assumptions.",
    },
];

export default function ExplainPage() {
    return (
        <main className="page-shell plain-page">
            <section className="hero-card guide-hero">
                <div className="hero-meta">
                    <div className="eyebrow">Guide</div>
                    <div className="hero-links">
                        <a href="/" className="repo-link">
                            Back to demo
                        </a>

                    </div>
                </div>

                <h1>In plain terms: What the model does</h1>
                <p className="hero-copy">
                    It helps answer a simple store question: how much stock should we prepare
                    for the next few days? The model looks at recent demand, predicts likely
                    future demand, and checks which restocking rule would have worked best.
                </p>

                <div className="guide-flow">
                    <div className="flow-step">
                        <span>1</span>
                        <strong>Look back</strong>
                        <p>Use the last 28 days of observed product demand.</p>
                    </div>
                    <div className="flow-step">
                        <span>2</span>
                        <strong>Forecast</strong>
                        <p>Predict demand for the next 7 days.</p>
                    </div>
                    <div className="flow-step">
                        <span>3</span>
                        <strong>Decide</strong>
                        <p>Choose a restocking policy from the forecast.</p>
                    </div>
                    <div className="flow-step">
                        <span>4</span>
                        <strong>Evaluate</strong>
                        <p>Compare the decision with what actually happened.</p>
                    </div>
                </div>
            </section>

            <section className="policies-card guide-story">
                <p>
                    Imagine a shelf in a grocery store. Every day, customers buy some number
                    of units from that shelf. Those daily numbers form a time series.
                </p>
                <p>
                    The model receives the last 28 daily numbers and predicts the next 7. A
                    point forecast gives one estimate, while the quantile forecast gives low,
                    middle and high scenarios.
                </p>
                <p>
                    The forecast is then used as a restocking decision. If we stock too little,
                    customers may not get the product. If we stock too much, inventory is left
                    over. The demo compares policies using stockout rate, fill rate and total
                    cost.
                </p>
            </section>

            <DictionaryGuide entries={dictionaryEntries} />
        </main>
    );
}
