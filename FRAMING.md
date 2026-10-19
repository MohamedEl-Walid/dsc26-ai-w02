## Decision

Rawaj's analysts need to decide which transfers to review for possible mule activity. The score supports a review queue; it does not establish fraud or justify an automatic account block. First audit the extract for duplicate transfers, missing labels, invalid timestamps and features that reveal an investigation's outcome. Confirm that every feature exists at the time a transfer would be scored.

## Metric

Use precision at the team's review capacity to estimate how much of the queue merits investigation, and track recall so a small queue does not hide missed cases. Select a threshold on validation data using total expected cost: false positives × cost_fp + false negatives × cost_fn, matching slide 11 of the W2 lesson. Dina must supply the costs and daily review capacity. Report the chosen threshold's confusion matrix and precision/recall. Keep the final test period separate from threshold selection.

## Baseline

Compare against flagging nothing and a simple existing rule or current queue if Rawaj supplies one. A chronological split better reflects future deployment than a shuffled split, and all copies of a transfer must remain in the same partition. Audit and remove leakage before fitting any model. The supplied task asks for the audit and score evaluation functions, so this repository does not claim to have trained a fraud model or measured performance on the missing sample extract.

## Classical ML or LLM

Start with rules and a conventional tabular model after data validation. Structured transfer attributes, explicit labels and a review threshold fit that approach. An LLM is unnecessary for computing the audit or binary metrics, adds cost and failure modes, and should not receive private transfer records. Consider language models only for a separately justified text task with appropriate privacy controls and an evaluation showing added value.
