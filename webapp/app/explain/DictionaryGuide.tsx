"use client";

import { useState } from "react";

type DictionaryEntry = {
    term: string;
    short: string;
    detail: string;
};

export default function DictionaryGuide({ entries }: { entries: DictionaryEntry[] }) {
    const [isOpen, setIsOpen] = useState(false);
    const [activeTerm, setActiveTerm] = useState<string | null>(null);

    return (
        <section className="policies-card dictionary-guide">
            <button
                type="button"
                className="dictionary-toggle"
                onClick={() => setIsOpen((current) => !current)}
                aria-expanded={isOpen}
            >
                <span>
                    <span className="section-title">Dictionary guide</span>
                    <span className="dictionary-subtitle">
                        Open if you want plain-language definitions of the key terms.
                    </span>
                </span>
                <span className={`chevron${isOpen ? " open" : ""}`} aria-hidden="true">
                    ↓
                </span>
            </button>

            {isOpen ? (
                <div className="dictionary-table" role="table">
                    <div className="dictionary-head" role="row">
                        <span role="columnheader">Term</span>
                        <span role="columnheader">Short meaning</span>
                    </div>

                    {entries.map((entry) => {
                        const entryIsOpen = activeTerm === entry.term;

                        return (
                            <div key={entry.term} className="dictionary-row-group">
                                <button
                                    type="button"
                                    className="dictionary-row"
                                    onClick={() =>
                                        setActiveTerm(entryIsOpen ? null : entry.term)
                                    }
                                    aria-expanded={entryIsOpen}
                                >
                                    <span className="dictionary-term">{entry.term}</span>
                                    <span className="dictionary-short">{entry.short}</span>
                                    <span
                                        className={`row-arrow${entryIsOpen ? " open" : ""}`}
                                        aria-hidden="true"
                                    >
                                        ↓
                                    </span>
                                </button>

                                {entryIsOpen ? (
                                    <div className="dictionary-detail">{entry.detail}</div>
                                ) : null}
                            </div>
                        );
                    })}
                </div>
            ) : null}
        </section>
    );
}
