# The staged fit's fixture (Phase 4.F spec D9), shared by test/config/test_deck_staged.jl (ST1 and
# ST2, the fast tier) and test/slow/test_deck_staged.jl (ST3 and ST4, the slow tier):
# examples/two_channel.toml's own deck, cut to 24 wavelengths per camera, 16 and 15 recorded slices
# and 4 bins, its two streaks synthesized from two_channel_problem with its own seeds. The staged
# table takes T_D from the IAW stage, as the example's procedure does.

include(joinpath(@__DIR__, "..", "..", "examples", "problems.jl"))

const _ST_CUT = (nlambda_epw=24, nlambda_iaw=24, nt_epw=16, nt_iaw=15, nbins=4)
const _ST_P = two_channel_problem(; _ST_CUT...)
const _ST_STAGED = "procedure = \"staged\"\nstaged = { stage_procedure = \"lbfgs\", start_from = { T_D = \"iaw\" } }"

_st_path(p::AbstractString) = replace(p, "\\" => "/")

function _st_write(dir::AbstractString, name::AbstractString, text::AbstractString)
    path = joinpath(dir, name)
    write(path, text)
    return path
end

# The two-channel deck text on this fixture's streaks: `optimizer` replaces procedure = "lbfgs".
function _st_text(dir::AbstractString; optimizer::AbstractString=_ST_STAGED, nbins::Integer=4)
    text = read(joinpath(@__DIR__, "..", "..", "examples", "two_channel.toml"), String)
    text = replace(text, "runs/two_channel_epw_streak.jld2" => _st_path(joinpath(dir, "epw.jld2")))
    text = replace(text, "runs/two_channel_iaw_streak.jld2" => _st_path(joinpath(dir, "iaw.jld2")))
    text = replace(text, "nbins = 32" => "nbins = $nbins")
    text = replace(text, "dir = \"runs/two_channel/joint\"" => "dir = \"$(_st_path(joinpath(dir, "out")))\"")
    return replace(text, "procedure = \"lbfgs\"" => optimizer)
end

function _st_streaks(dir::AbstractString)
    write_streak(joinpath(dir, "epw.jld2"), synthesize_channel(_ST_P, :epw); source="test")
    write_streak(joinpath(dir, "iaw.jld2"), synthesize_channel(_ST_P, :iaw); source="test")
    return nothing
end
