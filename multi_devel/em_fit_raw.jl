using LinearAlgebra, QuadGK, StatsBase, Distributions, Statistics
import Base: rand, +, /, -
import Distributions: cdf, ccdf

mutable struct MAPHDist
    α::Adjoint{Float64, Vector{Float64}}
    T::Matrix{Float64}
    T0::Matrix{Float64}
end

"""

Returns parameters of a two phase hyper-exponential fitting a mean and an SCV.

"""


function hyper_exp_fit(mean::Float64, scv::Float64)
    scv < 1.0 && error("SCV must be greater than 1")
    μ1 = 1/(scv+1)
    p = (scv-1)/(scv+1+2/μ1^2-4/μ1)
    μ2 = 2*(1-μ1)/(2-μ1*(scv+1))
    α = zeros(2)'
    α[1] = 1-p
    α[2] = p

    T = zeros(2,2)
    T[1,1] = -1/μ1
    T[2,2] = -(1-p)/(1-2*p)

    return (α, (1/mean)*T)
end

 

"""

Returns parameters of a hypo-exponential (generalized erlang) dist which is a sum of n exponentials with the last one different

"""

function hypo_exp_fit(mean::Float64,scv::Float64)

    scv ≥ 1.0 && error("SCV must be less than 1")

    n = Int(ceil(1/scv))

    ν1 = n/(1+sqrt((n-1)*(n*scv-1)))
    ν2 = ν1*(n-1)/(ν1-1)

    α = zeros(n)'
    α[1] = 1
    T = zeros(n,n)
    T[1,1] = -ν1
    T[1,2] = ν1

    for i = 2:(n-1)
        T[i,i] = -ν2
        T[i,i+1] = ν2
    end

    T[n,n] = -ν2

    return (α, (1/mean)*T) 

end


function scv(data::Vector{Float64})
    return (std(data)/mean(data))^2
end 

"""

Create an MAPHDist of dimensions pxq where q is the length of `probs`, `means`, and `scvs` and p is specified.

This tries to do a best "moment fit" for the probability of absorbitions, means, and scvs

"""
function MAPHDist(p::Int, probs::Vector{Float64}, means::Vector{Float64}, scvs::Vector{Float64})
    q = length(probs)
    length(means) != q && error("Dimension mismatch")
    length(scvs) != q && error("Dimension mismatch")

    πhat_order = sortperm(probs)
    reverse_order = sortperm(πhat_order)
    sorted_πhat = probs[πhat_order]
    sorted_scvs = scvs[πhat_order]
    sorted_means = means[πhat_order]
    # num_phases = zeros(q)

    # for i = 1:q
    #     if sorted_scvs[i] ≥ 1
    #         num_phases[i] = 2
    #     else # sorted_scvs[i] <1
    #         num_phases[i] = ceil(1/sorted_scvs[i])
    #     end
    # end

    num_phases = [sorted_scvs[i] ≥ 1 ? 2 : ceil(1/sorted_scvs[i]) for i in 1:q]

    required_phases = sum(num_phases)
    # selected_cases= [required_phases - sum(num_phases[1:k]) - p for k=1:q]

    
    K = findfirst((x)->x ≤ p, [required_phases - sum(num_phases[1:k]) for k=1:q])

    # scvs_required = sorted_scvs[K+1:q]
    # means_required = sorted_means[K+1:q]
    # πhat_required = sorted_πhat[K+1:q]

    # truncated_q = length(πhat_required)


    dist = []
    for k = 1:q
        if sorted_scvs[k]≥1
            push!(dist,hyper_exp_fit(sorted_means[k],sorted_scvs[k]))
        end
        if sorted_scvs[k]<1
            push!(dist,hypo_exp_fit(sorted_means[k],sorted_scvs[k]))
        end
    end

    for i = 1:K
        dist[i] = (ones(1,1),ones(1,1)).*eps()
    end

    
    reversed_dist = dist[reverse_order]

    reversed_α = (probs./sum(sorted_πhat[K+1:q])).*[reshape(reversed_dist[i][1],length(reversed_dist[i][1])) for i = 1:q]
    α = reduce(vcat,reversed_α)'
    
    reversed_T = (probs./sum(sorted_πhat[K+1:q])).*[reversed_dist[i][2] for i = 1:q]

    T = cat(reversed_T...,dims = (1,2))

    reversed_T0 = -(probs./sum(sorted_πhat[K+1:q])).*[sum(reversed_dist[i][2],dims=2) for i = 1:q]

    T0 = cat(reversed_T0...,dims=(1,2))

    display(α)

    display(T)
    display(T0)

    return MAPHDist(α,T,T0)
end


# function MAPHInit(p::Int=1,q::Int=1)
#     α = rand(p)
#     α = (α/sum(α))'
#     T = rand(1:0.01:20,p,p)
#     T0 = rand(1:0.01:20,p,q)
#     for i = 1:p
#         temp = T[i,i]
#         T[i,i]= -sum(T[i,:])-sum(T0[i,:])+temp
#     end
#     return MAPHDist(α,T,T0)
# end

# MAPHDist(;p::Int=1,q::Int=1) = MAPHInit(p::Int=1,q::Int=1)
# MAPHDist(;p::Int=1,q::Int=1) = MAPHDist(
#                         Vector{Float64}(undef,p)', 
#                         Matrix{Float64}(undef,p,p), 
#                         Matrix{Float64}(undef,p,q)) 




# ccdf(maph::MAPHDist, y::Float64, a::Int) = exp(maph.α*maph.T*y) 
# cdf = 1-ccdf
# V_j///

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
        return (y = t, a = state)
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
/(ss1::MAPHSufficientStats,ss2::MAPHSufficientStats) = MAPHSufficientStats(ss1.B ./ ss2.B, ss1.Z ./  ss2.Z, ss1.N ./  ss2.N)
-(ss1::MAPHSufficientStats, ss2::MAPHSufficientStats) = MAPHSufficientStats(ss1.B-ss2.B, ss1.Z-ss2.Z, ss1.N - ss2.N)



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


    # display(PT)
    # display(PT0)
    # display(PA)
    # display(sum(PA))


    EB(y::Float64, i::Int, j::Int) = maph.α[i] * b(y,j)[i]*PA[j]/ (maph.α*b(y,j))
    EZ(y::Float64, i::Int, j::Int) = c(y,i,j)[i]*PA[j]/(maph.α*b(y,j))
    # ENT(y::Float64,i::Int,j::Int) = i+q != j ?  maph.T[i,:].*c(y,i,j)*PA[j]/(maph.α*b(y,j)) : zeros(p)
    # ENT2(y::Float64,i::Int,j::Int) =  maph.T[i,:].*c(y,i,j)*PA[j]/(maph.α*b(y,j))

    ENT(y::Float64,i::Int,k::Int,j::Int) = i !=k ? maph.T[i,:].*c(y,i,j)*PA[j]/(maph.α*b(y,j)) : zeros(p)

    # ENA(y::Float64,i::Int,j::Int) = a(y)[i]*maph.T0[:,j][i]/(maph.α*b(y,j))
    ENA(y::Float64,i::Int,j::Int) = PA[j]*a(y)[i]*maph.T0[i,j]/(maph.α*b(y,j))

    stats.B = [sum([EB(observation.y, i, j) for j = 1:q]) for i =1:p]
    stats.Z = [sum([EZ(observation.y,i,j) for j =1:q]) for i = 1:p]


    for i = 1:p
        for k = (q+1):(q+p)
            V = sum([ENT(observation.y,i,k-q,j) for j in 1:q])
            stats.N[i,k] = V[k-q]
        end

   
        for j = 1:q
            stats.N[i,j] = ENA(observation.y,i,j)
        end

    end





    # for i= 1:p
    #     V = sum([ENT(observation.y,i,j) for j in 1:q])
    #     for k = q+1:q+p
    #         stats.N[i,k] = V[k-q]
    #     end

    #     for j = 1:q
    #         stats.N[i,j] = ENA(observation.y,i,j)
    #     end
    # end
    
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

function stats_to_dist(maph::MAPHDist,ss::MAPHSufficientStats)::MAPHDist
    p,q = model_size(maph)
    α = ss.B'
    T = ss.N[:,(q+1):(q+p)]./ss.Z
    T0 = ss.N[:,1:q]./ss.Z
    for i = 1:p
        temp = T[i,i]
        T[i,i]= -sum(T[i,:])-sum(T0[i,:])+temp
    end
    return MAPHDist(α,T,T0)
end
"""
Fits ... QQQQ
"""
function fit!(maph::MAPHDist,data::MAPHObsData)::MAPHDist
    #EM Loop

    p,q = model_size(maph)


    for k in 1:100
        ss = sufficient_stats(data[1],maph)
        for i in 2:10^2
            ss = ss+sufficient_stats(data[i],maph)
        end
        ss = ss/10^2
                maph = stats_to_dist(maph,ss)

    end


    return maph
end



"""
Returns an array of data filtered according to absorbing state being the index of the array.
"""
function absorb_filter_data(data, maph::MAPHDist)
    p, q = model_size(maph)
    filter_data = []

    for i = 1:q
        temp_data = filter(data) do obs first(obs).a == i end
        push!(filter_data,temp_data)
    end    
    
    return filter_data
end


function time_filter_data(data, n::Int64)

    max_time = maximum(((d)->first(d).y).(data) )#[data[i].y for i = 1:length(data)])
    time_vec = Array(LinRange(0, max_time, n)) #QQQQ - cleanup later

    filter_data = []

    for i = 1:(n-1)
        temp_data = filter(data) do obs first(obs).y ≥ time_vec[i] && first(obs).y < time_vec[i+1] end 
        push!(filter_data,temp_data)
    end    

    return(filter_data)
end


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
    @show test_stats
end

# test_example()

function test_example2()
    Λ₄, λ45, λ54, Λ₅ = 5, 2, 7, 10
    μ41, μ42, μ43, μ51, μ52, μ53 = 1, 1, 1, 1, 1, 1 
    T_example = [-Λ₄ λ45; λ54 -Λ₅]
    T0_example = [μ41 μ42 μ43; μ51 μ52 μ53]

    maph = MAPHDist([0.5,0.5]', T_example, T0_example)
   
    data = []
    full_trace =[]
    
    println("starting generating data")
    for i in 1:2*10^4
        times, states = rand(maph, full_trace = true) 
        push!(full_trace, (times,states))
        push!(data, (observation_from_full_traj(times,states),i))
        i % 10^5 == 0 && print(".")
    end
    println("\nfinished generating data")

    absorb = absorb_filter_data(data, maph)
    time_bin = time_filter_data(absorb[1], 1000)

    #loop over all bins
    println("\n start initialization...")
    for i in 1:length(time_bin)
        ss_i = MAPHSufficientStats[]
        for trace in full_trace[last.(time_bin[i])]
            ss = sufficient_stat_from_trajectory(maph, trace[1], trace[2])
            push!(ss_i, ss)
        end
        
        if !isempty(ss_i)
            mean_observed_ss = mean(ss_i)
            obs = first(data[last(last.(time_bin[i]))])
            computed_ss = sufficient_stats(obs, maph)
            errs_N = (mean_observed_ss.N - computed_ss.N) ./ computed_ss.N #./ computed_ss
            # @show obs
            # sufficient_stats()
        end
    end
    
    ab1 = filter(x->x.a==1,first.(data))
    ab2 = filter(x->x.a==2,first.(data))
    ab3 = filter(x->x.a==3,first.(data))
    
    probs = [length(ab1),length(ab2),length(ab3)]./length(data)
    means = [mean(first.(ab1)),mean(first.(ab2)),mean(first.(ab3))]
    scvs = [scv(first.(ab1)),scv(first.(ab2)),scv(first.(ab3))]

    dist = MAPHDist(10,probs,means,scvs)
    println("finish initialization")

    println("starting simulations")
    fit!(dist,first.(data))
    println("\nfinished simulations")
    
    return dist

    # @show first(data)
    # # @show m[1]
    # @show maximum(data.y)
    # @show maximum[data[i].y for i = 1:length(data)]
end

# test_example2();

# filter(ret) do obs obs.y ≥ 0.0 && obs.y < 0.005 && obs.a == 1 end 

# a = (filter(ret) do obs obs.a ==1 end);


#QQQQ - An example with a deterministic path....
function test_example3()
    
    ϵ = 0.00000001

    T_example = [-(1.0+4ϵ) 1.0-4ϵ ϵ 
                ϵ -(1.0 + 4ϵ) 1.0-4ϵ
                ϵ ϵ -(1 + 4ϵ) ]

    T0_example = [ϵ ϵ ϵ; 
                  ϵ ϵ ϵ;
                  1-4ϵ ϵ ϵ]

    
    T_example2 = [-(1.0+3ϵ) 1.0 ϵ
                 ϵ -(1.0+3ϵ) 1.0
                 ϵ ϵ -(1+3ϵ)]
    T0_example2 = [ϵ ϵ;
                   ϵ ϵ;
                   1-3ϵ ϵ]   

    display(sum(T0_example2[:,1]))
    maph = MAPHDist([0.5,0.5, 0.0]',T_example, T0_example)

    obs = (y=100.0, a=2)
    sufficient_stats(obs, maph)
end

# ss = test_example2()



function test_example4()
    Λ₄, λ45, λ54, Λ₅ = 5, 2, 7, 10
    μ41, μ42, μ43, μ51, μ52, μ53 = 1, 1, 1, 1, 1, 1 
    T_example = [-Λ₄ λ45; λ54 -Λ₅]
    T0_example = [μ41 μ42 μ43; μ51 μ52 μ53]

    maph = MAPHDist([0.5,0.5]', T_example, T0_example)

    println("starting simulations")
    data = [rand(maph) for _ in 1:10^2]
    

    p,q = model_size(maph)
    est_maph = MAPHInit(p,q)
    # est_maph = MAPHDist(;model_size(maph)...)
    @assert model_size(est_maph) == model_size(maph)

    #fit! gets an MAPHDist object for two possible reasons:
        #Reason #1 (always) - to know p and q.
        #Reason #2 (sometimes) - to have a starting guess. QQQQ - later give it a flag to say 
    fit!(est_maph,data)

    println("\nfinished simulations")
end

#ss=test_example4()

# a, T = hypo_exp_fit(0.23)


# @show T

out = test_example2()


density(y::Float64,dd::MAPHDist) = -dd.α*inv(dd.T)*(1-exp(dd.T*y))*dd.T0
