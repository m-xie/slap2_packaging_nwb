# SLAP2 NWB packaging

Packages synchronized SLAP2 fluorescence, visual stimuli, running, and eye
tracking into NWB, with associated QC and processing provenance.

## Soma ROI traces

When a source-bearing plane has `user_rois/labels` equal to `soma`
(case-insensitive, ignoring surrounding whitespace), its user-drawn soma ROIs
are packaged separately from extracted sources:

- `processing/ophys/ImageSegmentation/SomaPlaneSegmentation_DMD1` stores the
	projected user ROI masks, z extents, original labels, and zero-based user ROI indices.
- `processing/ophys/SomaFluorescence_DMD1` contains `DMD1_soma_F_green` and
	`DMD1_soma_F_red`, plus `DMD1_soma_Fsvd_green` and `DMD1_soma_Fsvd_red` when
	`Fsvd` is available. Only recorded channels are written; DMD names vary by plane.

These are the original `user_rois/F` and `user_rois/Fsvd` values, not source
`F0` or derived dF/F. They retain fluorescence NaNs and share the plane's HARP
timestamps, leading/trailing trial exclusions, and removal of samples with
non-finite timestamps. Existing source traces and QC selection are unchanged.
Absent user ROIs or labels other than `soma` add no soma output.

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
| `movie_frame_timestamps` | Float64 onset estimates in seconds relative to the first recorded SLAP2 DO0 pulse, matching the shared normalized HARP clock. Startup removal does not change this origin. Unsupported events have `NaN` timestamps. |
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

The movie timing QC figure includes a scatter plot of consecutive matched
photodiode changes: display-frame differences on the x-axis and analog-detected
time differences in milliseconds on the y-axis. These use matched photodiode
anchors, not interpolated movie timestamps. Intervals may span unmatched
transitions; invalid or non-increasing pairs are omitted without bridging them.

**Precision caveat:** timestamps are photodiode-aligned onset estimates, not
independent optical measurements of every movie frame. Logger/render ordering,
display scanout, ADC sampling, and edge-matching ambiguity can limit accuracy.
The existing session-level p95 affine residual gate is 40 ms; passing it does
not establish sub-frame precision. Zero residual at an interpolation anchor is
not a timing-accuracy measurement. Validate logger event semantics and optical
timing before interpreting these counters as exact MP4-frame onsets.

#### Continuous SLAP2 synchronization

For **Random Natural Movies**, SLAP2 is always treated as one continuous raw
acquisition. Source-extraction chunks are concatenated independently for each
DMD; different chunk counts do not imply different acquisition trials. Raw
files must describe one acquisition and logical `TRIAL1`, either unchunked or
contiguous cycle chunks starting at offset zero. The `-TRIAL<number>` token may
be omitted; these files all belong to trial 1 (for example,
`<acquisition>_DMD1.dat` or `<acquisition>_DMD1-CYCLE-000000.dat`). Before synchronization, an
initial DO0/DO1 pair is rejected when its positive duration is less than 100 ms
and less than 10% of the median later paired duration (the existing startup
heuristic). At least two complete pairs are required for this startup heuristic.
After trimming, exactly one finite start and zero or one finite stop must remain.
If present, the stop must follow the start. Missing starts or extra markers raise
an error. DO1 is optional: imaging timestamps come from DI3, and stimulus trial
labeling uses an open interval when no stop is recorded.

The first recorded DO0 remains time zero, even if that marker is discarded.
Startup removal does not re-zero retained timestamps, recording onset, or the
absolute `time_reference`. Stimulus, imaging, running, and eye data continue to
share this original HARP origin. No DI3 pulses or analog samples are removed.
Shared startup cleanup removes associated early DO2 events for both formats;
Random Natural Movies does not use DO2. Data preceding the first recorded DO0
retain negative timestamps; the retained acquisition start may be positive.
The retained markers define the acquisition interval without further slicing.

The first detected DI3 pulse is **primary path cycle 1**, the next is cycle 2,
and so on. There is no gap segmentation, leading-pulse reconciliation,
pre-DO0 pulse filtering, or effective-lines-per-cycle calculation in this mode.
DMD1 is always the primary, even when it has no extracted sources; secondary
source-bearing DMDs do not replace it. Missing DMD1 or its raw files is an error.
A DI3 pulse requires an observed low-to-high transition. An initially high
state always produces a warning but is not counted as a pulse.

Exact `linesPerCycle` and complete recorded cycle counts are obtained through
the pinned **SLAP2_Utils** raw-header parser (`load_file_header_v2`, also used by
its `DataFile` class), with read-only memory mapping. Full fluorescence data and
metadata are not loaded merely to count cycles. Chunk lengths are checked
against filename offsets, counts are summed, and changing cycle lengths fail.
Both planes retain their original global, 1-based scan-line coordinates.
Primary cycle boundary lines are `1 + cycle_index * lines_per_cycle`; each
fluorescence sample is interpolated at its actual scan-line position between
the corresponding DI3 boundaries. Missing early extracted samples do not shift
the clock. Secondary samples use that same primary scanner-line map, checked
against their own raw path's recorded line count.
Scan-line indices beyond that count produce a warning and receive `NaN`
timestamps, even if covered by the shared clock. Before constructing NWB
fluorescence series, samples with non-finite timestamps are removed from both
timestamps and fluorescence arrays. Unfiltered timestamps remain available in
memory for synchronization QC; unsupported samples are not stored in NWB.

For `C` complete primary cycles and `P` detected DI3 pulses:

- Require `C >= P - 1`, allowing one extra pulse for the start of a final cycle
	not logged completely by SLAP2. More extra pulses raise an error.
- When `P == C + 1`, the extra pulse supplies the measured end of cycle `C`.
- Otherwise, with at least two pulses, only the last observed cycle's end is
	estimated from the mean measured cycle period, with a warning. No later
	cycle starts are fabricated; samples in unobserved cycles receive `NaN`.
- With one pulse, only its exact cycle-start line can be timed. No pulses fail.
- A high DI3 state at recording onset, or the first rising edge within **1 ms**
	of onset, warns that SLAP2 may have started before HARP began recording.
	Recording onset uses the earliest recorded analog/digital timestamp, not
	normalized DO0 time zero. Because digital logging is event-driven, a first
	high record well after HARP onset does not trigger this onset warning, but
	still triggers the initial-high warning and is not counted. All observed
	low-to-high pulses are retained; an unknown missing prefix is not guessed
	or corrected.

Per-DMD JSON reports in the synchronization QC directory include raw path
counts, primary identity, pulse counts, onset warnings, final-cycle policy,
and unsupported/stored sample counts. Imaging alignment anchors are not saved
in NWB. Legacy logger formats retain their trial-based timing algorithms and
initial-high behavior: an initial high state contributes a synthetic pulse at
the first sample's timestamp without an initial-high warning. DMD1 remains
primary, and non-finite fluorescence timestamps are excluded from NWB as above.

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