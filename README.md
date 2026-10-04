# SLAP2 NWB packaging

Packages synchronized SLAP2 fluorescence, visual stimuli, running, and eye
tracking into NWB, with associated QC and processing provenance.

## Random Natural Movies

Select **Random Natural Movies** in the capsule's **Logger Format** setting
(`--logger_format`). The other logger-format selections retain their existing
stimulus parsing, timing fallback, table names, and QC behavior.

The adapter requires a stimulus table with `TextureName`, `TrialType`,
`TrialDuration`, `ExtentX`, and `ExtentY`, and a BonVision logger with `Frame`,
`Timestamp`, and `Value`. With the unchanged legacy stimulus-table-pattern
default, the new format falls back to a unique filename containing
`stim_table`. An explicit custom pattern is honored; ambiguous matches fail.

### Output intervals

- **stimulus_blocks**: every observed table row in its original order, with
	all input columns retained. `stimulus_table_row` is the zero-based source row
	and matches this table's NWB id. For the attached session there are 55 rows:
	35 movie presentations and 20 grating blocks.
- **gratings**: individual `GratingStart-*` / `GratingEnd-*` presentations,
	including blanks (180 rows for the attached session). `stimulus_table_row`
	identifies the parent block; `grating_in_block` is zero-based. The logger
	value 359 is retained as `logger_orientation`, but represents `is_blank=True`
	with `Orientation=NaN`, not a 359-degree grating.
	Shared `SpatialFrequency`, `TemporalFrequency`, `DiameterX`, `DiameterY`,
	`X`, `Y`, and `Contrast` are read from the acquisition metadata's
	`stimulus_epochs[].code.parameters.StimulusParameters`. Column descriptions
	include units and source fields. The circular `GratingDiameter` supplies
	both diameter columns. Blank trials have zero contrast; other properties
	retain the configured values. Nominal grating duration and delay are not
	added. Missing stimulus metadata produces a warning; incomplete or ambiguous
	settings are rejected rather than guessed.

Both tables include aligned `start_time` and `stop_time`, measured `Duration`,
source `start_frame` and `stop_frame`, and `slap2_trial_idx`. No movie identity
is present in the logger: identities come from the ordered stimulus table,
validated against the movie/grating sequence and repeated movie frame counts.
Unobserved table rows are omitted, not assigned guessed times. The original
row identifiers are preserved even when only a prefix can be synchronized.

### Timing and validation

#### Per-movie frame timing

Each movie row in `stimulus_blocks` now contains four equal-length NWB ragged
arrays (non-movie rows contain typed empty arrays):

| Column | Meaning |
| --- | --- |
| `movie_frame_timestamps` | Float64 onset estimates in seconds relative to the first SLAP2 DO0 pulse, matching the existing normalized HARP clock. Unsupported events have `NaN` timestamps. |
| `movie_frame_numbers` | Original 1-based `MovieFrame-N` logger counters, reset for each presentation. |
| `movie_display_frames` | Global logger `Frame` coordinates for those events. |
| `movie_frame_timing_status` | UInt8 codes: **0** anchored, **1** interpolated, **2** endpoint-extrapolated, **3** unsupported. |

`movie_url` links known movie textures to the supplied commit-pinned GitHub
assets; it is empty for non-movie rows or unknown textures. Movie counters are
preserved as logged, not claimed to be independently verified decoded MP4
indices. Each event describes one logged content-frame presentation, not every
monitor refresh. No frames are synthesized from nominal FPS or trial duration.

Frame timestamps use the **same matched photodiode anchors and piecewise-linear
map as interval boundaries**, computed once per session. Exact matched anchors
receive status 0. Interpolation is unsupported when the surrounding anchor gap
exceeds **three times the median logger photodiode-transition spacing**. This
threshold is based on all logged transitions, not the potentially sparse matched
subset; it is a coverage heuristic, not an accuracy guarantee. The Python
`synchronize_presentations()` API permits an explicit positive
`maximum_interpolation_gap_frames` override. Endpoint extrapolation remains
limited to two display frames and receives status 2.

For interrupted recordings, existing interval recovery still omits whole blocks
starting beyond usable coverage. **All events within retained blocks are kept**,
including events beyond a clipped stop: those events receive status 3 and `NaN`,
never a timestamp clamped to the last edge. `movie_frame_count` counts preserved
logged events, not just finite timestamps. Timing provenance reports logged,
stored, omitted, and per-status counts. Filter by finite timestamps and the
desired quality codes before fine-alignment analysis; the arrays remain paired
by index when filtering.

Alignment anchors are used in memory for synchronization and the timing QC
plot, but are **not saved in NWB**. The per-frame arrays remain in
`stimulus_blocks`. Policy, clock reference, and QC summaries are included in
external processing provenance under `stimulus_timing.movie_frame_timing`;
the original HARP clock offset is recorded when available.

**Precision caveat:** timestamps are photodiode-aligned onset estimates, not
independent optical measurements of every movie frame. Logger/render ordering,
display scanout, ADC sampling, and edge-matching ambiguity can limit accuracy.
The existing session-level p95 affine residual gate is 40 ms; passing it does
not establish sub-frame precision. Zero residual at an interpolation anchor is
not a timing-accuracy measurement. Validate logger event semantics and optical
timing before interpreting these counters as exact MP4-frame onsets.

#### Interval timing and recovery

- `MovieFrame-1` identifies the first displayed content frame. Counters must
	progress continuously and restart at 1 for every movie, including adjacent
	movies. The **global display Frame**, not the content counter or logger
	Timestamp, is mapped to physical HARP photodiode edges using the existing
	alignment algorithm and quality gates.
	For this format the pre-stimulus patch is assumed low: an initial state at
	`STARTSLAP` is inserted as an edge only when high. A low state following an
	actually observed high state remains a real falling edge. Legacy-format
	first-state handling is unchanged.
- Movie stops use the next movie's first frame or the final session `END`
	event. The logger has no movie-off event before gratings: those offsets are
	estimated as the last content frame's display frame plus the **observed
	median content-frame spacing**, including the final frame's display period.
	This limitation is explicit in each block's `stop_frame_source` and in
	timing provenance; it is not an independently measured movie-off edge.
- `TrialDuration` remains nominal metadata and is **never used for alignment**.
- Each complete grating block must contain eight distinct directions and one blank.
	Table-level grating bounds span the first logged start through the last
	logged end. Unlogged leading/trailing blanks are excluded; inter-grating
	gaps within a block are included. Individual grating bounds use their
	paired events directly.
- `END` and `EndFrame` are optional; either can identify the final boundary.
	If both exist they must agree. If neither exists, the parser warns and
	recovers the recorded prefix rather than requiring a completed session.
	A logger ending before the table produces a warning naming the first
	unobserved row and the number omitted.
- An interrupted movie or grating block is retained with `is_partial=True`.
	Completed individual grating pairs remain valid; a final unmatched
	`GratingStart` is retained with a censored stop at the last observed display
	frame. Missing or malformed events *inside* the recorded sequence, duplicate
	directions, extra events after the table, and out-of-order events still fail
	because row identity cannot be safely recovered by guessing.
- Single-frame movies are retained with a warning. Without a known playback
	boundary, only their observed prefix is stored. If the logger ends on that
	onset, `start_time == stop_time` and `Duration == 0`: this is an onset-only
	censored observation, **not** a claim that the stimulus lasted zero seconds.
	Partial stops are lower bounds, not measured stimulus offsets, and must not
	be used as complete trial durations. `stop_frame_source` identifies the
	boundary policy. Shortened final repeats warn rather than fail.
- Playback offsets are not inferred from DO2; there is no new-format DO2 fallback.
- Boundaries are interpolated between matched physical photodiode edges.
	At the recording endpoints only, at most **two display frames** of extension
	are permitted using the locally fitted HARP slope. This handles an `END`
	immediately after the last photodiode transition without silently clamping
	its time. For interrupted sessions, a longer uncovered tail is clipped to
	the last matched anchor and marked partial; starts beyond that anchor are
	omitted with warnings, preserving earlier alignable data. Completed sessions
	still fail on unsupported coverage. Any extension, partial intervals, omitted
	rows, and recovery warnings are recorded in provenance. A recording without
	enough photodiode transitions for a reliable fit still cannot be synchronized.

The analog photodiode threshold remains the midpoint of the acquisition-window
10th and 90th percentiles. Two-cluster estimation may help when either state
occupies less than 10% of the recording, but is not enabled automatically:
clustering can split baseline noise or outliers into artificial high/low states.
It should be evaluated with occupancy, separation, and alignment-residual checks
before changing established synchronization behavior.

### QC

Stimulus-specific QC (Zebra repeats, receptive fields, stimulus tuning, and
orientation tuning) is skipped for this format for now. General fluorescence,
running, eye, and synchronization processing/QC remain unchanged. Existing
formats continue to run the original stimulus-specific QC.

Random Natural Movies packaging additionally writes a movie-frame timing PNG
inside the QC synchronization directory. It shows within-block inter-frame
intervals without bridging missing timestamps, anchor affine residuals
(explicitly **not** accuracy estimates), and global-frame coverage with anchor
spacing and unsupported/extrapolated events.

## Tests

Tests use `unittest` with the code directory on `PYTHONPATH`; discover tests
under [code/tests](code/tests). The Random Natural Movies tests include
synthetic malformed logs, playback boundary handling, NWB validation and
round-trip, legacy isolation, and an optional read-only parser check against
the attached session.