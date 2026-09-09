# CLAUDE.md

OTIS.jl is a Julia package for forward modelling and inverting optical Thomson scattering
spectra, aimed at streaked (time-resolved) data inverted with regularization along the time
axis. The name expands to **O**ptical **T**homson **I**nversion **S**uite. The reference text is
Sheffield, Froula, Glenzer and Luhmann, *Plasma Scattering of Electromagnetic Radiation*, 2nd
ed. (`context/sheffield2011.pdf`; cite it as Sheffield with a section or equation number). The
user's own paper, `context/foo2026.pdf`, describes the method and the test problems but may
contain typos; where it and Sheffield conflict, Sheffield is the source of truth. The design
motivation that distinguishes it: the velocity distribution `f(v)` is an **input, not an
assumption**. Laser-heated plasmas are routinely non-Maxwellian, and a Maxwellian fit to such
data returns a temperature that is not any moment of the real distribution. A distribution is
therefore a dimensionless *shape*, anything evaluable at `x = (v-u)/vth`, with density, flow and
temperature on the species that carries it, so that super-Gaussian, kappa and expansion shapes
are first-class and the shape itself can eventually be recovered from the data instead of
fitted away.

This repository is a fresh start. An earlier implementation was archived separately and none of
it is in this tree; do not assume a subsystem exists because a document describes it, check
`src/`. The work is organized as a chain of assistants at three levels. Read this file first: it
says what your level means and how levels hand work to each other.

## Levels

Every assistant is assigned a level, 1, 2 or 3, by the user at the start of the session. If you
have not been told your level, ask before doing anything else.

**Keep your level in your context.** State it in your first reply ("Working as Level 2"), at
the top of every hand-off summary, and again when resuming after a context summary. If it has
dropped out of your context, ask rather than guess.

Level N's job is to work with the user to produce instructions and prompts for Level N+1, and
to carry out Level N−1's instructions as written.

| Level | Owns | Produces | Does not |
|---|---|---|---|
| 1 | `dev/roadmap.md`, `dev/open_questions.md`, `dev/conventions.md`; the broad strokes of the package | Decisions recorded in those three documents; `dev/phaseN_brief.md`, the brief that starts a Level 2 session | Write package code; write phase specs |
| 2 | One phase of the roadmap at a time | `dev/phaseN_instructions.md` (the spec), `dev/phaseN_prompt.md` (the prompt that starts the Level 3 session), and `dev/phaseN_overall_summary.md` (the running record of what goes back to Level 1) | Write package code (except under the three-strikes rule below); reopen decisions in the Level 1 documents |
| 3 | The code | `src/`, `test/`, `docs/derivations/`, one commit per step, a hand-off summary in `dev/reports/` | Edit anything in `dev/` outside `dev/reports/`; build ahead of the spec; reopen decisions |

## Where the project stands

The three Level 1 documents exist (written 2026-09-06) and are binding. Code arrives phase by
phase: the status line at the top of each phase in `dev/roadmap.md` says what is built, and
nothing else does. The archived implementation is not a reference; what survived of its
conventions is in `dev/conventions.md`.

## Lower-level instructions are gospel

Instructions from a lower-numbered level are implemented as written: tolerances, file names,
signatures, formulas, step order, scope. If an instruction looks wrong, impossible or
improvable:

1. Stop that piece of work.
2. Tell the user what the instruction says, what you found, and what you propose instead.
3. Wait for approval. Do not implement the deviation while waiting, and do not quietly
   implement it afterward.
4. Once approved, record the deviation with the date where the next reader will see it: in
   your hand-off summary, and, for the level that owns the document, in the document itself.

"Seemingly justified" is not an exemption. A test that fails at its stated tolerance is a
finding to report, not a tolerance to relax. A cleaner API than the spec's is a proposal, not a
change.

Findings flow upward too. Anything a level discovers that should change a lower level's
document goes into its hand-off summary as a **[propagate]** item, and the lower level edits its
own document. No level edits a document it does not own.

### Level 2 keeps a running summary for Level 1

**Level 2 maintains `dev/phaseN_overall_summary.md` for the phase it owns**, and keeps it
current from the moment the phase starts until the phase is signed off. It is the one place
Level 1 has to read to see what its own documents still need, so that it never has to
reconstruct that from a spec, a review and three hand-off files.

It accumulates across the whole phase — the brief, the spec, Level 3's reports, Level 2's code
review, and every remediation round — rather than being written once at the end. Level 2 updates
it whenever a new finding appears, not just at hand-off. It holds:

- **What needs a Level 1 decision**, each item naming the document and section it belongs in,
  quoting what that document says now, and proposing replacement text. An item is
  decision-ready: Level 1 should be able to accept it verbatim, not re-derive it.
- **A log of what Level 1 has already folded in**, so nothing is done twice and Level 2 can
  verify it landed.
- **Measurements worth quoting** in the Level 1 documents — accuracies, timings, costs — with
  the machine and package versions they were taken on.
- **What is still pending** from a round in flight.

Mark each item open, decided or folded, and date it. Where an item is already settled by the
user, say so and say it is not to be reopened. This does not replace the per-session hand-back
prompt Level 2 writes when a phase's findings are ready to discuss; the summary is the standing
record behind it.

## Reviews

Levels 1 and 2 may be asked to review code that Level 3 has written. A review is read-only:
the reviewer reports findings to the user and does not edit `src/` or `test/`. Level 2 reviews
against its spec (does the code do what the instructions say, are the tests at the stated
tolerances, was anything built ahead or left out). Level 1 reviews against the roadmap and
`dev/conventions.md` (architecture, interfaces, conventions compliance, whether approved
deviations were recorded). Every finding names the file and line and says which document it
violates; the user decides whether Level 3 fixes the code or the owning level changes the
document.

## Three strikes

If Level 3 has failed three times to bring a piece of code in line with Level 2's spec, the
initial prompt counting as the first attempt and each remediation prompt as one more, Level 2
may fix that piece itself. The permission is per item, not per phase: Level 2 edits only what
the failed item covers in `src/`, `test/` and `docs/derivations/`, runs the test suite, commits
with the step named and "fixed by Level 2" in the message, and reports in its hand-off summary
what it changed and why Level 3 could not. Everything else stays with Level 3, and a Level 2
fix is not a license to tidy nearby code.

## Writing for the next level

Higher-numbered levels will usually be weaker models with less context. Instructions written for
them must be:

- **Self-contained.** Everything needed is in the instructions or named by exact path. The
  reader should not have to infer a formula from Sheffield or a convention from the codebase;
  formulas are written out in the code's normalization, not cited by equation number alone.
- **Ordered and bounded.** Steps in order, each with a definition of done (tests with their
  tolerances), and an explicit list of what is out of scope for this hand-off.
- **Explicit about the traps.** A ranked pitfall list, and for every tolerance whether it is
  deliberately loose or deliberately tight, and why. Units, signs and normalizations are where
  this kind of code goes wrong; say which convention each formula is written in.
- **Decision-free for the reader.** Where the lower-level documents leave a gap, the writing
  level closes it with a dated decision in the spec, marked [propagate] if it belongs in a
  Level 1 document. The reader should never have to choose between interpretations.
- **Closed with a required report.** Say exactly what the hand-off summary must contain:
  tests added and the tolerances they actually run at, deviations and their approvals,
  [propagate] items, and any naming or convention ambiguities met on the way.

Once written, `dev/phase0_instructions.md` and `dev/phase0_prompt.md` are the reference
examples of the form. Until then, the same form for a different package is at
`C:\Users\bfoo\Documents\GitHub\JANUS.jl\dev\phase0_instructions.md` and
`C:\Users\bfoo\Documents\GitHub\JANUS.jl\dev\phase0_prompt.md`; copy the structure, not the
content.

## Briefs

Level 1 starts a Level 2 session with `dev/phaseN_brief.md`, and writes it only when every
question in that phase's section of `dev/open_questions.md` is decided: not provisional, not
decided in outline, not waiting on the user. If one is still open, decide it with the user
first. Every brief says that the roadmap beyond the phase in hand is expected to change, so
the spec must not build anything that depends on a later phase's choices; a phase defines only
what its own tests exercise.

## Repository map

- `src/`, `test/`, `docs/derivations/`: the package, tracked by git. `test/` mirrors `src/`.
- `dev/`: planning documents and phase hand-offs. Gitignored. Level 3 reads it and edits
  nothing in it except `dev/reports/`.
- `dev/reports/`: Level 3's hand-off summaries. The summary for phase N goes in
  `dev/reports/phaseN_report.md`; a remediation round appends a dated section to the same file
  rather than starting a new one.
- `dev/phaseN_overall_summary.md`: Level 2's running record, for phase N, of everything that
  belongs in a Level 1 document — open items with proposed text, what is already folded in, and
  the measurements worth quoting. Kept current through the phase, not written once at the end.
- `context/`: reference material, currently `sheffield2011.pdf`, the Sheffield et al. textbook.
  Gitignored.
- `dev/conventions.md` is binding for every level: units and where they are converted,
  normalizations and sign conventions, naming rules, testing rules. Where it is silent, flag
  the gap rather than choose.

## Commands and git

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run it at the end of every step. One commit per roadmap step, with the step named in the
message (`Step 0.2: constants and unit conversions`). Never commit `dev/`, `context/` or
`Manifest.toml`.

## Environment

Windows 11, Julia 1.12.7; the package targets Julia 1.10. The Bash tool on this machine has no
coreutils (`sed`, `ls`, `cat`, `grep`, `head` all fail) and no `python`: use PowerShell or the
file tools. PowerShell's `-replace` and `-match` are case-insensitive unless written `-creplace`
and `-cmatch`. Julia sources here use Unicode identifiers (`φ`, `ω`, `α`); create and edit them
with the file tools, never with `Set-Content` or `Out-File`, which do not preserve UTF-8 reliably.

**If Julia starts crashing repeatedly, stop and await further instruction.** Julia and its
language server on this machine have crashed in bursts — most recently 2026-09-09, part-way
through Phase 1.SG Step 1.SG.4, where restarting VSCode cleared it. One crash is bad luck;
**three or four in a row with similar error messages is a broken environment, and the session
must stop trying.** Record the error message and what you were running, once. Then:

- **Do not keep reaching for another way to run it.** Retrying through `cmd`, through a different
  shell, through a script file instead of `-e`, with different flags, or in a fresh call is not
  diagnosis — it is the same failure again, and a burst of near-identical attempts buries the one
  error message the user needs in noise.
- **Do not try to fix the environment, and above all do not try repeatedly.** Clearing the
  precompile cache, editing `Project.toml` or `Manifest.toml`, re-instantiating, reinstalling,
  changing versions: each can leave the environment worse than you found it, and none of it is
  any level's job here. **One attempt at a fix is already one too many** unless the user has
  asked for it.
- **Carry on with whatever does not need Julia**, then **hand back and wait for instructions.**
  Say plainly what is done, what is blocked, and what the tree looks like. Do not press on into
  work that depends on the blocked part, and do not treat handing back early as a failure — it
  is the correct outcome, and it is far more useful than a long transcript of identical crashes.

Committing work whose tests you could not run is not an option; leave it uncommitted and describe
the state of the tree so the next session can pick it up.

**Never report a number you could not measure, or a test you could not run.** Leave it explicitly
blank and say why ("not measured — Julia crashed"). Do not estimate it, do not extrapolate it, and
do not carry an earlier session's figure forward as if it were this one's. A missing measurement
is something the user can act on; an invented one is not, because nobody downstream can tell it
from a real one. The same holds for a test: say that it did not run, never that it passed.
