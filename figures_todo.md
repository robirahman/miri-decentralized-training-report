# Figures TODO — next paper version

Suggestions for additional figures to add in the next revision of the report,
ordered roughly by payoff-per-effort. `#1` is the strongest pick; `#4`–`#7`
are "if-space-allows."

## 1. Cost vs. bandwidth (Appendix C, alongside or replacing Table 2)

A log–log curve with bandwidth on the x-axis (10 Mbps → 1 Gbps) and cost on
the y-axis, with the China-avg and US-avg points marked, plus a dashed
horizontal line at the centralized-training baseline cost. The new Table 2
numbers trace a beautiful elbow — cost barely moves from 1 Gbps down to
~100 Mbps ($12.9M → $30.7M), then explodes to $2.79B at 10 Mbps. A single
picture conveys "bandwidth is not a meaningful barrier until you fall below
typical residential broadband," which is one of the paper's most
policy-relevant findings and is currently buried in a table. This is my #1
pick — cleanest visual payoff for the fewest words.

## 2. η decomposition across the configurations in Table 1

A stacked bar chart (or horizontal bars) with one bar per Table 1 row and
segments for η_H, η_compression, η_replica, and η_activation. Would make
immediately legible that replica divergence dominates at mid-scale and that
activation compression takes over under PP for the 10²⁶ rows — claims
currently made in prose but hard for the reader to internalize at the speed
of reading. Fits inside Section 3.3 (Efficiency Model) or early in §4.

## 3. Cost multiplier from the 1,280-GB memory threshold

This is your key policy recommendation, and the current presentation
("roughly 50%" for 10²⁵, "approximately five times as many nodes") buries it
in one Appendix F paragraph. A two-panel figure — panel (a) cost vs.
compute target, with and without the memory cap; panel (b) node count vs.
compute target, with and without — makes the recommendation visual. Both
panels are 7 dots each (one per Table 1 row), easy to produce. This is the
figure reviewers will cite if they cite any.

## 4. DiLoCo schematic (Background)

A simple flowchart: N replicas each doing H inner steps on local data →
pseudo-gradient → all-reduce across replicas → synchronized parameter
update. Most ML readers know data-parallel training but not DiLoCo's
specifics; this saves 100 words of prose in §2 and helps readers who jump
straight to methodology. Low-information-density by itself, but high
pedagogical payoff. Good candidate if you have space after trimming prose;
skip if the 8-page body is tight.

## 5. η_replica vs. replica count, faceted by model size

Currently the paper asserts η_replica ranges from 0.15–0.90 and that "this
penalty scales inversely with model size" — visualizing the actual curves
(4–8 lines, one per model size from 50B to 310B) would make the
scaling-with-N argument visible. Good companion to the planned C_quality
vs. node-count plot. Goes in §3.3 or §4.

## 6. Training-time ceiling vs. regulatory stringency (Appendix A)

Plot T = 1 / (g_H + g_S + g_I) as a function of cumulative growth-rate
reduction on the x-axis, with your two operating points labeled (unregulated
≈ 4.5 months at 1.37×·3×·3.5× growth; treaty ≈ 740 days at 6%·50%·3%
growth). Makes the policy-paradoxical finding — stringent regulation
lengthens the max-useful training run — immediately visible. It's not a
showstopper but it shores up the T derivation that currently just drops a
formula and two parameter sets on the reader.

## 7. Countermeasure effectiveness matrix rendered as a heatmap

The table in Appendix F is already good, but a 6×2 grid (rows =
countermeasures, cols = non-state/state) with cells color-coded by the
High/Medium/Low rating, plus a "burden" glyph at the end of each row, lets
readers absorb the recommendation pattern in one glance. This is really
just a presentation upgrade of an existing table, so treat it as a
nice-to-have.
