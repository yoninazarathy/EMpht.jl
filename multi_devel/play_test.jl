using Pkg, Random, Distributions
cd(@__DIR__)
Pkg.activate(".")
@show pwd()
include("../src/EMpht_multi.jl")

Random.seed!(1)
sObs = Sample(obs=rand(Exponential(), 1_000))
ph1unif = empht(sObs, p=1, method=:unif)
