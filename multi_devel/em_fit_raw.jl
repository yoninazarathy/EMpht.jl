using LinearAlgebra, QuadGK, StatsBase, Distributions, Statistics
import Base: rand, +, /

mutable struct MAPHDist
    α::Adjoint{Float64, Vector{Float64}}
    T::Matrix{Float64}
    T0::Matrix{Float64}
end

function q_matrix(d::MAPHDist)::Matrix{Float64}
    p, q = model_size(d)
    return [zeros(q,p) zeros(q,q) ; d.T0 d.T]
end

function p_matrix(d::MAPHDist)::Matrix{Float64}
    p, q = model_size(d)
    PT = I-inv(Diagonal(d.T))*d.T
    PT0 = -inv(Diagonal(d.T))*d.T0
    return [I zeros(q,p); PT0 PT]
end

function rand(d::MAPHDist; full_trace = false)
    p, q = model_size(d)
    transient_states = (q+1):(q+p)
    all_states = 1:(q+p)
    Pjump = p_matrix(d)
    Λ = vcat(zeros(q), -diag(d.T))

    if full_trace
        states = Int[]
        sojourn_times = Float64[]
    end

    state = sample(transient_states,weights(d.α)) #initial state
    t = 0.0
    while state ∈ transient_states
        sojourn_time = rand(Exponential(1/Λ[state]))
        if full_trace
            push!(states,state)
            push!(sojourn_times,sojourn_time)
        end
        t += sojourn_time
        state = sample(all_states,weights(Pjump[state,:]))
    end

    if full_trace
        push!(states,state)
        return (sojourn_times, states)
    else
        return (t, state)
    end
end

"""
Create a named tuple observation.
"""
observation_from_full_traj(times::Vector{Float64}, states::Vector{Int64}) = (y = sum(times),a = last(states))


mutable struct MAPHSufficientStats
    B::Vector{Float64} #initial starts
    Z::Vector{Float64} #time spent
    N::Matrix{Float64} #transitions
    MAPHSufficientStats(B::Vector{Float64}, Z::Vector{Float64}, N::Matrix{Float64}) = new(B,Z,N)
    function MAPHSufficientStats(maph::MAPHDist) 
        p, q = model_size(maph)
        new(zeros(p), zeros(p), zeros(p,p+q))
    end
end

+(ss1::MAPHSufficientStats, ss2::MAPHSufficientStats) = MAPHSufficientStats(ss1.B+ss2.B, ss1.Z+ss2.Z, ss1.N+ ss2.N)
/(ss::MAPHSufficientStats,n::Real) = MAPHSufficientStats(ss.B/n,ss.Z/n,ss.N/n)

function sufficient_stat_from_trajectory(d::MAPHDist, sojourn_times::Array{Float64}, states::Array{Int})::MAPHSufficientStats
    p,q = model_size(d)
    transient_states = (q+1):(q+p)

    
    ss =  MAPHSufficientStats(d)

    for s = 1:p
        if states[1] == transient_states[s]
            ss.B[s] +=1
        end
    end

    for i = 1:(length(states)-1)
        ss.Z[states[i]-q] += sojourn_times[i]
        ss.N[states[i]-q, states[i+1]] += 1
    end



    #QQQQ do stuff
    return ss
end



model_size(maph::MAPHDist) = (p = size(maph.T,1), q = size(maph.T0,2)) #transient is p and abosrbing is q

SingleObs = NamedTuple{(:y, :a), Tuple{Float64, Int64}}
MAPHObsData = Vector{SingleObs}


#temp
function very_crude_c_solver(y::Float64,i::Int,j::Int,maph::MAPHDist)
    quadgk(u -> (maph.α*exp(maph.T*u))[i]*exp(maph.T*(y-u))*maph.T0[:,j] , 0, y, rtol=1e-8) |> first
end

function sufficient_stats(observation::SingleObs, maph::MAPHDist; c_solver = very_crude_c_solver)::MAPHSufficientStats
    stats = MAPHSufficientStats(maph)

    p, q = model_size(maph)

    a(y::Float64) = maph.α*exp(maph.T*y)
    b(y::Float64,j::Int) = exp(maph.T*y)*maph.T0[:,j]
    c(y::Float64,i::Int,j::Int) = very_crude_c_solver(y,i,j,maph)

    D = Diagonal(maph.T)
    PT = I-inv(Diagonal(maph.T))*maph.T
    PT0 = -inv(Diagonal(maph.T))*maph.T0
    A = inv(I-PT)*PT0
    PA = maph.α*A

    EB(y::Float64, i::Int, j::Int) = maph.α[i] * b(y,j)[i]*PA[j]/ (maph.α*b(y,j))
    EZ(y::Float64, i::Int, j::Int) = c(y,i,j)[i]*PA[j]/(maph.α*b(y,j))
    ENT(y::Float64,i::Int,j::Int) = i+q != j ?  maph.T[i,:].*c(y,i,j)*PA[j]/(maph.α*b(y,j)) : zeros(p)
    ENA(y::Float64,i::Int,j::Int) = a(y)[i].*maph.T0[i,j]/(maph.α*b(y,j))

    stats.B = [sum([EB(observation.y, i, j) for j = 1:q]) for i =1:p]
    stats.Z = [sum([EZ(observation.y,i,j) for j =1:q]) for i = 1:p]

    for i= 1:p
        V = sum([ENT(observation.y,i,j) for j in 1:q])
        for k = q+1:q+p
            stats.N[i,k] = V[k-q]
        end

        for j = 1:q
            stats.N[i,j] = ENA(observation.y,i,j)
        end
    end
    
    return stats
end

sufficient_stats(data::MAPHObsData, maph::MAPHDist; c_solver = very_crude_c_solver)::MAPHSufficientStats = mean([sufficient_stats(d, maph) for d in data])


# mutable struct ProbabilityMatrix
#     P::Matrix{Float64} # transient to transient
#     P0::Matrix{Float64}#transient to absorbing
#     function ProbabilityMatrix(maph::MAPHDist)
#         p, q = model_size(maph)
#         new(zeros(p,p), zeros(p,q))
#     end
# end

# function update_probability_matrix(maph::MAPHDist, probmatrix::ProbabilityMatrix)
#     probmatrix.P = I-inv(Diagonal(maph.T))*maph.T
#     probmatrix.P0 = -inv(Diagonal(maph.T))*maph.T0
# end



function test_example()
    Λ₄, λ45, λ54, Λ₅ = 5, 2, 7, 10
    μ41, μ42, μ43, μ51, μ52, μ53 = 1, 1, 1, 1, 1, 1 
    T_example = [-Λ₄ λ45; λ54 -Λ₅]
    T0_example = [μ41 μ42 μ43; μ51 μ52 μ53]

    maph = MAPHDist([0.5,0.5]',T_example, T0_example)
    stats = MAPHSufficientStats(maph)
    # @show maph
    # @show model_size(maph)

    data = [(y=2.3,a=2),(y=5.32,a=1),(y=15.32,a=2)]
    # update_sufficient_stats(maph, data,stats)

    # @show stats

    # QQQQ update_sufficient_stats(maph,data,stats)

    test_stats = MAPHSufficientStats(maph)

    for _ in 1:10^5
        times, states = rand(maph, full_trace = true) 
        ss = sufficient_stat_from_trajectory(maph, times, states)
        test_stats.N += ss.N
        test_stats.Z += ss.Z
        test_stats.B += ss.B
    end

    test_stats.B = test_stats.B/sum(test_stats.B)
    test_stats.Z = test_stats.Z/sum(test_stats.Z)
    test_stats.N = test_stats.N/sum(test_stats.N)

    # @show(stats.N,test_stats.N)
    
    @show stats.Z #- test_stats.N
    @show test_stats.Z
end

# test_example()


function test_example2()
    Λ₄, λ45, λ54, Λ₅ = 5, 2, 7, 10
    μ41, μ42, μ43, μ51, μ52, μ53 = 1, 1, 1, 1, 1, 1 
    T_example = [-Λ₄ λ45; λ54 -Λ₅]
    T0_example = [μ41 μ42 μ43; μ51 μ52 μ53]

    maph = MAPHDist([0.5,0.5]',T_example, T0_example)
   
    data = SingleObs[]

    for _ in 1:10^5
        times, states = rand(maph, full_trace = true) 
        push!(data,observation_from_full_traj(times,states))
        # ss = sufficient_stat_from_trajectory(maph, times, states)
        # test_stats.N += ss.N
        # test_stats.Z += ss.Z
        # test_stats.B += ss.B
    end
    # [d for d in data]
    # [sufficient_stats(d, maph) for d in data]
    # stats = MAPHSufficientStats(maph)
    # update_sufficient_stats(maph,data
    # data[1
    data
    # sufficient_stats(data,maph)
end

ret = test_example2()

filter(ret) do obs obs.y ≥ 0.0 && obs.y < 0.005 && obs.a == 1 end 
