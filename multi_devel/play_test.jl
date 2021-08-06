using Pkg
cd(@__DIR__)
Pkg.activate(".")
@show pwd()
include("../src/EMpht_multi.jl")
