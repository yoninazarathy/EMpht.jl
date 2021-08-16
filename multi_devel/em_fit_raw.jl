using LinearAlgebra, QuadGK

mutable struct MAPHDist
    α::Adjoint{Float64, Vector{Float64}}
    T::Matrix{Float64}
    T0::Matrix{Float64}
end

# function doobGillespie(Q::Matrix{Float64},init_p::Vector{Float64},absorbing_states::Set{Int64})
#     n = size(Q)[1]
#     Pjump  = (Q-diagm(0 => diag(Q)))./-diag(Q)
#     lamVec = -diag(Q)
#     state  = sample(1:n,weights(initProb))
#     sojournTime = rand(Exponential(1/lamVec[state]))
#     t = 0.0
#     while t + sojournTime < T
#         t += sojournTime
#         state = sample(1:n,weights(Pjump[state,:]))
#         sojournTime = rand(Exponential(1/lamVec[state]))
#     end
#     return state
# end


model_size(maph::MAPHDist) = (p = size(maph.T,1), q = size(maph.T0,2)) #transient is p and abosrbing is q

SingleObs = NamedTuple{(:y, :a), Tuple{Float64, Int64}}

MAPHObsData = Vector{SingleObs}

mutable struct SufficientStats
    B::Vector{Float64} #initial starts
    Z::Vector{Float64} #time spent
    N::Matrix{Float64} #transitions
    function SufficientStats(maph::MAPHDist) 
        p, q = model_size(maph)
        new(zeros(p), zeros(p), zeros(p,p+q))
    end
end

#temp
function very_crude_c_solver(y::Float64,i::Int,j::Int,maph::MAPHDist)
    quadgk(u -> (maph.α*exp(maph.T*u))[i]*exp(maph.T*(y-u))*maph.T[:,j] , 0, y, rtol=1e-8) |> first
end

function update_sufficient_stats(maph::MAPHDist, data::MAPHObsData; c_solver = very_crude_c_solver)
    # @show maph
    # @show data

    #QQQQ - Here start to add code to update the sufficents stats

    a(y::Float64) = maph.α*exp(maph.T*y)
    b(y::Float64,j::Int) = exp(maph.T*y)*maph.T0[:,j]
    c(y::Float64,i::Int,j::Int) = very_crude_c_solver(y,i,j,maph)
    # @show a(3.3)
    # @show b(3.3,1)
    # @show c(3.3,1,2)

    EB(y::Float64, i::Int, j::Int) = maph.α[i] * b(y,j)[i] / (maph.α*b(y,j))
    @show EB(3.3,1,1)
end

mutable struct ProbabilityMatrix
    P::Matrix{Float64} # transient to transient
    P0::Matrix{Float64}#transient to absorbing
    function ProbabilityMatrix(maph::MAPHDist)
        p, q = model_size(maph)
        new(zeros(p,p), zeros(p,q))
    end
end

function update_probability_matrix(maph::MAPHDist, probmatrix::ProbabilityMatrix)
    probmatrix.P = I-inv(Diagonal(maph.T))*maph.T
    probmatrix.P0 = -inv(Diagonal(maph.T))*maph.T0
end


function test_example()
    Λ₄, λ45, λ54, Λ₅ = 3, 2, 7, 10
    μ41, μ42, μ43, μ51, μ52, μ53 = 1, 1, 1, 1, 1, 1 
    T_example = [-Λ₄ λ45; λ54 -Λ₅]
    T0_example = [μ41 μ42 μ43; μ51 μ52 μ53]

    maph = MAPHDist([0.5,0.5]',T_example, T0_example)
    # @show maph
    probmatrix = ProbabilityMatrix(maph)
    # @show model_size(maph)
    # @show SufficientStats(maph)

    data = [(y=2.3,a=2),(y=5.32,a=1),(y=15.32,a=2)]
    update_sufficient_stats(maph, data)
    update_probability_matrix(maph,probmatrix)
end

test_example();

