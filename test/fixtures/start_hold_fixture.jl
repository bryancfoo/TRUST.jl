# The start and hold tables' fixture (Phase 4.I spec D13), shared by test/data/test_start.jl (SF),
# test/inversion/test_held_series.jl (HD) and test/config/test_deck_staged.jl's ST5. It is written
# against test/fixtures/staged_fixture.jl, which the including file includes first: the two-channel
# deck and its four-bin streaks (bins at 2.633, 3.52, 4.28 and 5.167 ns). Added here: an
# trust-start-1 series on a coarser axis that spans those bins (_SH_*), the deck's EPW channel alone as
# a one-channel deck, the TOML edit that adds the two tables to a deck's text, and the linear
# interpolation written out as the oracle for build_problem's.

import TOML

const _SH_AXIS_NS = collect(range(2.1, 5.7; length=7))
const _SH_SERIES = Dict("T_e" => collect(range(500.0, 900.0; length=7)),
    "logn" => collect(range(19.3, 19.8; length=7)), "m_e" => [3.5])
const _SH_UNITS = Dict("T_e" => "T_eV")

# The fixture's start file in `dir` (or `name`), written by write_start in ns.
function _sh_start_file(dir::AbstractString; name::AbstractString="start.jld2", values=_SH_SERIES,
    units=_SH_UNITS, axis_ns=_SH_AXIS_NS)
    return write_start(joinpath(dir, name); axis_ns=axis_ns, values=values, units=units, source="test")
end

# The staged fixture's EPW channel alone (single_channel_deck_text), with `optimizer`.
function _sh_epw_text(dir::AbstractString; optimizer::AbstractString="procedure = \"lbfgs\"")
    text = single_channel_deck_text(_st_text(dir; optimizer=optimizer), "epw")
    return replace(text, "runs/two_channel/epw" => _st_path(joinpath(dir, "out")))
end

# `text` parsed, edited in place by `f(raw)` and printed again, sorted (the form
# single_channel_deck_text prints, in which a table's text is not stable enough to edit as text).
function _sh_edit(f, text::AbstractString)
    raw = TOML.parse(text)
    f(raw)
    buf = IOBuffer()
    TOML.print(buf, raw; sorted=true)
    return String(take!(buf))
end

# `text` with a [fit.start] and/or a [fit.hold] table added, each given as (file, names).
function _sh_tables(text::AbstractString; start=nothing, hold=nothing)
    return _sh_edit(text) do raw
        start === nothing || (raw["fit"]["start"] = Dict("file" => _st_path(start[1]), "variables" => start[2]))
        hold === nothing || (raw["fit"]["hold"] = Dict("file" => _st_path(hold[1]), "variables" => hold[2]))
    end
end

# The file's value at `t` (s): a one-value entry, the value itself at an axis point, else the linear
# formula on the bracketing pair -- the oracle for build_problem's interpolation (SF4, HD5).
function _sh_interp(axis_s::AbstractVector, v::AbstractVector, t::Real)
    length(v) == 1 && return v[1]
    k = findlast(x -> x <= t, axis_s)
    axis_s[k] == t && return v[k]
    return v[k] + (t - axis_s[k]) / (axis_s[k+1] - axis_s[k]) * (v[k+1] - v[k])
end
