using LinearAlgebra

mutable struct MAPHParameters
    T::Matrix{Float64}
    T0::Matrix{Float64}
end

model_size(maph::MAPHParameters) = (p = size(maph.T,1), q = size(maph.T0,2)) #transient is p and abosrbing is q

SingleObs = NamedTuple{(:y, :a), Tuple{Float64, Int64}}

MAPHObsData = Vector{SingleObs}

mutable struct SufficientStats
    B::Vector{Float64} #initial starts
    Z::Vector{Float64} #time spent
    N::Matrix{Float64} #transitions
    function SufficientStats(maph::MAPHParameters) 
        p, q = model_size(maph)
        new(zeros(p), zeros(p), zeros(p,p+q))
    end
end

function update_sufficent_stats(maph::MAPHParameters, data::MAPHObsData)
    @show maph
    @show data

    #QQQQ - Here start to add code to update the sufficents stats
end

mutable struct ProbabilityMatrix
    P::Matrix{Float64} # transient to transient
    P0::Matrix{Float64}#transient to absorbing
    function ProbabilityMatrix(maph::MAPHParameters)
        p, q = model_size(maph)
        new(zeros(p,p), zeros(p,q))
    end
end

function update_probability_matrix(maph::MAPHParameters, probmatrix::ProbabilityMatrix)
    probmatrix.P = I-inv(Diagonal(maph.T))*maph.T
    probmatrix.P0 = -inv(Diagonal(maph.T))*maph.T0
end


function test_example()
    Λ₄, λ45, λ54, Λ₅ = 3, 2, 7, 10
    μ41, μ42, μ43, μ51, μ52, μ53 = 1, 1, 1, 1, 1, 1 
    T_example = [-Λ₄ λ45; λ54 -Λ₅]
    T0_example = [μ41 μ42 μ43; μ51 μ52 μ53]

    maph = MAPHParameters(T_example, T0_example)
    probmatrix = ProbabilityMatrix(maph)
    @show model_size(maph)
    @show SufficientStats(maph)

    data = [(y=2.3,a=2),(y=5.32,a=1),(y=15.32,a=2)]
    update_sufficent_stats(maph, data)
    update_probability_matrix(maph,probmatrix)
end

test_example()

