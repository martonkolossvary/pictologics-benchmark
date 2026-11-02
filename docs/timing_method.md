# Timing method

The benchmark measures calculation time only.
[`configs/benchmark/main.json`](../configs/benchmark/main.json) fixes the
timing policy, and [`bench/benchmark_contract.py`](../bench/benchmark_contract.py)
rejects any other policy. [`bench/run.py`](../bench/run.py) is the
controller. [`bench/adapters/base.py`](../bench/adapters/base.py) holds the
clock. The inputs are in [`benchmark_design.md`](benchmark_design.md), and the
timed groups are in [`feature_contract.md`](feature_contract.md).

This document uses these terms:

- A **task** is one adapter, one case, one timed group and one repeat.
- A **call** is one calculation of all the families that the adapter supports
  in the group.
- A **repeat** is one fresh adapter process.
- A **branch** is the scope of a skip decision (see [Skip rules](#skip-rules)).

## The timing boundary

| Outside the clock | Inside the clock |
| --- | --- |
| Process start, Python start-up and package import | ROI voxel extraction, for every adapter |
| The Pictologics `warmup_jit()` call, which compiles its Numba kernels before the adapter reads the files | Matrix, mesh, run, zone and neighbourhood construction |
| File reading and hash checks | Feature formula evaluation |
| Mask binarization and the ROI crop | The PyRadiomics shape class construction, which calculates the mesh, the diameters and the principal axes |
| Resegmentation and discretisation (the stored representations need none) | Cache sharing between features inside one adapter call, as the package does it |
| Native object preparation: the PyRadiomics first-order and texture classes, the MIRP feature objects, the Z-Rad `RoiData` and the MEDimage volumes | |
| Clearing the Z-Rad local-intensity cache and every cached table of the Pictologics feature modules, before and after each call | |
| Result conversion, value checks and JSON output | |

Wall time comes from `time.perf_counter_ns` and CPU time from
`time.process_time_ns`. The CPU time counts all threads of the adapter
process.

Before the clock, every adapter crops its inputs to the ROI bounding box plus
one voxel on each side. For local intensity, the margin is the sphere radius
of 6.2 mm, rounded up to whole voxels (`roi_crop` in
[`bench/adapters/protocol.py`](../bench/adapters/protocol.py)). Z-Rad uses its
own `RoiCropper` with the same margin. The PyRadiomics shape class pads its
mask by one voxel itself, so the adapter crops that mask with no margin. Each
adapter then keeps only the cropped arrays. Thus all adapters calculate on the
same voxels. Each package still prepares its own data structures before the
first call, so memory values above the worker-ready level do not measure the
same data (see [Memory](#memory)).

## One task

1. The controller checks the power state, the memory preflight and the skip
   decisions.
2. It starts a fresh adapter process in the isolated environment of that
   adapter.
3. The adapter imports its package and reads the files. Pictologics runs
   `warmup_jit()` before it reads the files. The adapter then sends the
   `inputs_loaded` event.
4. The adapter crops and prepares its inputs.
5. It runs one untimed warmup call and discards the result.
6. It runs three timed calls. Preparation before each call and finalization
   after each call stay outside the clock.
7. It writes one JSON payload with every sample and every feature value.
8. The controller validates the payload, writes it atomically and commits it
   to the SQLite ledger.

Each task also records these times:

- `load_sec`: from the process start to `inputs_loaded`;
- `setup_sec`: from `inputs_loaded` to the first call;
- `warmup_duration_sec`: the warmup call;
- `jit_warmup_sec`: the Pictologics `warmup_jit()` call;
- the host wall time and the process CPU time of the whole task.

The controller fails a task when the payload does not pass its checks. The
checks include the adapter identity and version, the input hashes, the number
of warmup and timed calls, and the full event stream. The returned feature
count must equal the count in `main.json`, and every value must be finite. A
failed task has no timing value.

### Task order

- Each pillar runs in its own controller with its own ledger, in the order
  Pillar 1, Pillar 2, Pillar 3.
- Inside a pillar, the repeat is the outer loop. All tasks of repeat 1 finish
  before repeat 2 starts.
- Inside a repeat, cases run from the smallest to the largest image. For each
  case, the six groups run in contract order.
- For each case and group, the five adapters run in one order of a Williams
  design (`_williams_orders` in [`bench/run.py`](../bench/run.py)). With five
  adapters, the design has ten orders.
- The order number is `(case index + group index + repeat − 1) mod 10`. Thus
  ten successive cases of one group and repeat use all ten orders.
- Over the ten orders, each adapter takes each position twice, and runs
  directly after each other adapter twice. Thus no adapter always runs first,
  and no adapter always follows the same adapter. A heavy all-core task warms
  the computer equally before each adapter.
- The run order of each task (`ordinal`) and its start time go into the
  report records, so a later analysis can test order effects.
- The run adds no pause between tasks. The balanced order spreads the effect
  of the previous task equally over the adapters.

## Summary statistics

- The primary value of one process is the median of its three timed calls
  (`duration_sec` and `cpu_time_sec`).
- The payload also holds the minimum, mean, population standard deviation and
  maximum, and every raw sample, for wall time and CPU time.
- The controller recomputes each statistic from the raw samples. A mismatch
  fails the task.
- The three calls of one process show short-term timing noise. They are not
  independent replicates.
- Across the three processes, the reports use the median of the process
  medians, with the interquartile range ([`bench/report.py`](../bench/report.py)).
- A cell (one adapter, case and group) gets a median only when all its
  repeats are measured. A censored, failed or skipped repeat leaves the cell
  without a median, because the finished repeats alone would understate the
  time. The cell keeps `planned_repeats` and `timing_observations`.
- A runtime ratio divides an adapter's time by the Pictologics time for the
  same case, group and repeat. The reports give the median and interquartile
  range of the ratios, the number of matched pairs and the planned pairs.
  A cell with a missing pair gets no ratio summary.
- Figure 5 counts a timeout of either side: a package timeout gives a lower
  bound and a Pictologics timeout an upper bound ([`figures.md`](figures.md)).
- Each task record keeps every timed call (`duration_samples_sec`,
  `cpu_time_samples_sec`), so a later analysis can use a different statistic
  without a rerun. Disclose such a choice.

Each task also records these memory values:

- the peak RSS of the process tree;
- the peak memory in use (see [Memory](#memory));
- the worker-ready RSS, after the warmup and before the first timed call;
- the peak RSS during the timed calls, and its increase over the worker-ready
  RSS;
- the peak RSS of each call.

The reports and Figure 6 use the peak memory in use of the whole process tree.
This is the measure of the memory stop, and it counts everything that a
package needs. The increase over the worker-ready RSS is only a secondary
value: each package prepares a different amount of data before the first
call, and memory that the warmup used can stay with the process.

Each task also records the host load average at its start and end, and an
hourly speed canary. The canary times a fixed single-thread loop and flags a
host slowdown.

## Outcomes and censoring

| Status | Meaning | Timing value |
| --- | --- | --- |
| `measured` | The payload passed all checks | Yes |
| `timed_out_censored` | A call, the start-up or a gap between calls reached the time limit | No. A call timeout gives a lower bound. |
| `memory_exceeded_censored` | The memory in use went above the memory budget | No |
| `skipped_policy` | A prior timeout or memory stop skipped this task | No |
| `skipped_memory_preflight` | The memory estimate was above the budget, so the task did not start | No |
| `unsupported` | The adapter does not calculate this group | No |
| `failed` | The calculation or the payload checks failed | No |
| `interrupted` | The run stopped during the task. A resume runs it again. | No |

- No censored, skipped, unsupported or failed task gets a duration, a zero or
  an imputed value.
- The reports exclude these tasks from medians, ratios and fits. They show
  the counts and reasons as censoring information.
- A timeout or memory-stop record keeps the warmup time and the times of the
  completed calls as diagnostics. The task stays censored.
- A run with a failed, timed-out or memory-stopped task ends with the status
  `completed_with_failures`.

## Time limits

- The main run uses a limit of 1,800 s per call. The value comes from the
  launch policy in `workspace_manifest.json`.
- The limit applies to each call alone: the warmup and each timed call.
- The same limit bounds the start-up phase (process start to the first call),
  each gap between calls, and the time after the last call.
- A timeout record stores `stop_phase` (`startup`, `warmup`, `calculation` or
  `between_calls`), `stop_call_index`, the warmup time and the completed call
  times.
- It also stores `censor_lower_bound_sec`, the elapsed time of the call that
  reached the limit. A warmup timeout bounds the warmup. A start-up or
  between-call timeout has no bound.

## Memory

### Memory in use

[`bench/process_tree.py`](../bench/process_tree.py) measures the memory in
use of the adapter process and all its child processes:

| System | Measure |
| --- | --- |
| macOS | Physical footprint (`phys_footprint` from `proc_pid_rusage`) |
| Linux | `VmRSS` plus `VmSwap` from `/proc/<pid>/status` |
| Windows | Private bytes |

RSS leaves out compressed and swapped pages. Thus a process can use up the
memory while its RSS stays small. The memory stop uses the memory in use. The
controller still records RSS for the reports.

### Memory budget and stop

The memory budget is the smaller of 80 percent of the total memory and the
total memory minus 4 GiB. Only the total memory sets the budget, so a busy
moment on the host cannot change it. On the 48 GiB reference host
([`configs/benchmark/hosts/mac-m4pro-01.json`](../configs/benchmark/hosts/mac-m4pro-01.json)),
the budget is 38.4 GiB.

- The controller reads the memory in use every 0.05 s.
- When the value goes above the budget, the controller stops the process tree
  and records `memory_exceeded_censored`.
- On macOS and Linux, the operating system can kill a large process before
  the next reading. If the adapter ends by `SIGKILL` after it held at least
  half the budget, the controller also records a memory stop.
- A memory stop has no lower bound on the runtime.
- A QC warning flags a measured task whose peak RSS went above the free
  memory at launch, because swapping can slow such a task.

### Memory preflight

[`bench/benchmark_memory.py`](../bench/benchmark_memory.py) estimates the peak
memory before each task:

1. It takes the input bytes: the uncompressed size of the selected image plus
   the mask, from the NIfTI headers.
2. It makes a static estimate: 512 MiB plus 18 times the input bytes. The
   factor is the same for every adapter, so a guess never skips one package
   before another.
3. After one measured task of the same adapter, group and branch, it also
   makes an empirical estimate. The estimate is 1.10 × the largest
   worker-ready RSS + 1.5 × r × the input bytes. Here r is the memory increase
   per input byte of the largest prior input.
4. It uses the larger estimate. If that estimate is above the budget, the task
   gets `skipped_memory_preflight` and does not start.

The estimate is a launch-safety rule. The reports never show it as measured
memory. A preflight skip does not skip other tasks.

## Skip rules

A skip decision applies to one adapter, one group and one branch:

| Pillar | Branch |
| --- | --- |
| 1. Morphology | One noise level and one mask |
| 2. Whole anatomy | The A1 series |
| 3. Clinical cohort | One case |

- A timeout in the warmup or in a timed call skips that size in later repeats
  and every larger size in the same branch.
- A memory stop skips in the same way.
- A start-up or between-call timeout can come from a random stall. The first
  one in a branch censors only its own task. A second one in the same branch
  skips, as a call timeout does.
- The rules apply to all five adapters alike, Pictologics included. A skip
  never crosses to another adapter or group.
- A clinical branch holds one case, so clinical cases never share skips. A
  stop on one case still skips that case in later repeats.
- The launcher passes `--disable-speed-truncation`. Thus no task gets a skip
  because it is slow relative to another adapter.

The controller stores the cutoff one voxel below the image voxel count of the
stopped task. A later task of the same branch with more image voxels than the
cutoff gets `skipped_policy`, with the reason and the evidence task.

## Threads

The main contract sets `thread_policy` to `adapter_native_threads`. Each
adapter can then reach its fastest speed. The controller removes nine
variables from the adapter environment. Thus no thread limit from the shell
can leak in. The variables are `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`,
`NUMBA_NUM_THREADS`, `PICTOLOGICS_NUM_THREADS`,
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` and `BLIS_NUM_THREADS`.

- Pictologics has explicit parallel feature code in Numba. Other adapters can
  use several cores through the maths libraries under NumPy and SciPy.
- Pictologics 1.0.0 uses its fast cores by default: the performance cores on
  Apple silicon (10 of the 14 logical CPUs of an M4 Pro), and all logical CPUs
  on most other computers. Each Pictologics payload records this count
  (`package_initialization.threads`).
- CPU time beside wall time shows the parallel work of each call.
- The run specification stores the thread policy, the removed and set
  variables, and the logical CPU count. These values are part of the run
  fingerprint.
- Each adapter environment snapshot stores its package set and the output of
  `numpy.show_config()`, which names the linked BLAS and LAPACK. These are
  part of the fingerprint too.

The one-library-thread screen
([`configs/benchmark/screen_one_library_thread.json`](../configs/benchmark/screen_one_library_thread.json))
sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` and `BLIS_NUM_THREADS` to 1.
Numba and ITK keep their native threads, so the Pictologics parallel code
does not change. Together with the feasibility screen, it gives the time of
each task with and without the library threads. The main run always uses
native threads.

## Screens and the quick run

| Run | Contract | Cases | Warmup | Timed calls | Repeats | Limit per call | Scheduled records | Result folder |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Main run | [`main.json`](../configs/benchmark/main.json) | 283 | 1 | 3 | 3 | 1,800 s | 25,470 | `<machine-id>/<pillar>` |
| Feasibility screen | [`screen.json`](../configs/benchmark/screen.json) | 283 | 1 | 1 | 1 | 600 s | 8,490 | `<machine-id>/screen/<pillar>` |
| One-library-thread screen | [`screen_one_library_thread.json`](../configs/benchmark/screen_one_library_thread.json) | 283 | 1 | 1 | 1 | 600 s | 8,490 | `<machine-id>/screen_one_library_thread/<pillar>` |
| Quick run | [`screen.json`](../configs/benchmark/screen.json) | 12 | 1 | 1 | 1 | 300 s | 360 | `<machine-id>/quick/<pillar>` |

- The screens use the same timing boundary, cases, groups and adapters as
  the main run. Each has its own contract, run names and ledger.
- The feasibility screen shows which tasks finish, time out or stop on
  memory. Its report writes `task_outcomes.csv` with one row per task. Use it
  to estimate the time of the main run.
- The quick run checks a machine on 12 small cases
  (`QUICK_CASES` in [`scripts/launch_benchmark.py`](../scripts/launch_benchmark.py)):
  - Pillar 1: M1 at 32³ and 64³, at all three noise levels (6 cases).
  - Pillar 2: A1 at 32³, 48³ and 64³ (3 cases).
  - Pillar 3: the cases with a largest image edge of 136, 112 or 49
    (STS_019 CT, STS_019 MRI and STS_042 PET).
- The results of the screens and the quick run are not comparable with the
  main run. Never pool them with it.

[`scripts/launch_benchmark.py`](../scripts/launch_benchmark.py) prints the
commands by default. It calculates only with `--execute --confirm CALCULATE`.
Add `--screen`, `--screen --one-library-thread` or `--quick` to select a run.
[`reproduce.sh`](../reproduce.sh) and [`reproduce.ps1`](../reproduce.ps1) run
the full sequence with one command. Their `--screens` option runs both screens
before the main run. Their `--quick` option runs the quick run in place of the
main run.

## Stop and resume

- **Ctrl-C.** Press Ctrl-C once and wait for "Benchmark interrupted safely".
  The controller stops the adapter process tree, records the task as
  `interrupted` and checkpoints the ledger. The launcher then starts no
  further pillar.
- **Closed terminal or `SIGTERM`.** The controller treats `SIGHUP` and
  `SIGTERM` as Ctrl-C. After `SIGHUP`, it sends its output to the null
  device, so the clean stop can finish. The launcher passes `SIGTERM` to the
  controller.
- **Hard kill, crash or power loss.** The ledger keeps the open task as
  `running`. At the next start, the controller marks it `interrupted`. It also
  stops any adapter that the dead controller left on the inputs of the run,
  before the first task. On Windows, a Job Object ends every adapter process
  when the controller exits for any reason.
- **Power check.** With a host profile, the controller checks the power state
  before each task: AC power, low-power mode off and a sleep-prevention
  assertion. The controller holds this assertion itself: `caffeinate` on
  macOS, `systemd-inhibit` on Linux and `SetThreadExecutionState` on Windows.
  If a check fails, it stops before the task with exit code 75.

To resume after any stop:

1. Correct the cause, for example connect AC power.
2. Run the identical command again.

- The launcher always passes `--resume`. Each pillar resumes from its own
  ledger, and a finished pillar does no new calculation.
- An interrupted task runs again as a complete fresh-process task. The new
  attempt replaces the record, and the attempt table keeps every attempt.
- Measured, censored, skipped, unsupported and failed tasks stay as they are.
  The controller checks the checksum of every committed payload again.
- Resume rejects a different run fingerprint. The fingerprint holds the
  inputs and their absolute paths, the task plan, the time limit and all
  policies. It also holds the source hashes, the adapter environments, and
  the host identity with its kernel release.
- The controller also checks the source and input hashes before and after
  each task, so a live edit stops the run.
- A different `--result-root` or `--machine-id` starts a new run in a new
  folder.

To add repeats 4 and 5 after a complete main run, pass `--repeats 5`. The
launcher adds `--extend-repeats`. The existing tasks keep their identities,
values and skip decisions. A smaller repeat count fails.

## What you may and may not conclude

You may:

- compare adapters on matched tasks, with the same case, group, stored
  representation and repeat;
- model scaling and memory within each pillar, as
  [`benchmark_design.md`](benchmark_design.md) describes;
- report the counts of timeouts, memory stops and skips, and the timeout lower
  bounds, as outcomes;
- use the two screens to show how much the thread policy changes each task.

You may not:

- divide runtime by the feature count, or make a features-per-second or
  coverage-adjusted speed score;
- claim equal feature work, because each adapter calculates its own native
  feature set;
- add group times to make a full-extraction time;
- read a group time as an end-to-end time, because file reading, mask
  preparation and discretisation are outside the clock;
- give a censored, skipped, unsupported or failed task a value;
- pool the results of the screens or the quick run with the main run;
- pool the three noise levels of Pillar 1, or the two synthetic pillars;
- treat results from two hosts as one software environment. The run
  fingerprint ties each result to its host and environment.

`main.json` records these limits under `interpretation`:
`equal_feature_work_claim`, `per_feature_runtime_valid` and
`coverage_adjusted_runtime_valid` are all `false`.
