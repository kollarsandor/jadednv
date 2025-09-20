# =======================================================================

#!/usr/bin/env julia
# Enhanced Self-Contained Mathematical Operations Module
# Combines existing functionality with advanced operations from uploaded system

module EnhancedSelfContainedMath

using LinearAlgebra
using Statistics
using SparseArrays
using Base.Threads

export euclidean_distance, matrix_multiply, vectorized_softmax, softmax, rmsd,
       optimized_matmul, to_sparse_distance_matrix, accelerated_distance_matrix,
       exists, default, log, divisible_by, compact, l2norm, max_neg_value,
       to_device, dict_to_device, symmetrize, masked_average, exclusive_cumsum,
       pad, unpad, reshape_for_attention

# Basic utility functions
function exists(v)
    return !isnothing(v) && !ismissing(v)
end

function default(v, d)
    return exists(v) ? v : d
end

# Enhanced Euclidean distance without external dependencies
function euclidean_distance(p1::Tuple{Float64,Float64,Float64}, p2::Tuple{Float64,Float64,Float64})
    sqrt((p1[1]-p2[1])^2 + (p1[2]-p2[2])^2 + (p1[3]-p2[3])^2)
end

function euclidean_distance(p1::AbstractVector, p2::AbstractVector)
    @assert length(p1) == length(p2) "Vector dimensions must match"
    sqrt(sum((p1[i] - p2[i])^2 for i in 1:length(p1)))
end

# Enhanced matrix operations
function matrix_multiply(A::Matrix{Float64}, B::Matrix{Float64})
    m, n = size(A)
    n2, p = size(B)
    @assert n == n2 "Matrix dimensions don't match: $(size(A)) × $(size(B))"

    C = zeros(Float64, m, p)
    @inbounds for i in 1:m, j in 1:p, k in 1:n
        C[i,j] += A[i,k] * B[k,j]
    end
    return C
end

# Fixed Vectorized Softmax with proper type support
function vectorized_softmax(x::AbstractArray{T}; dims=-1) where T<:Real
    if dims == -1
        dims = ndims(x)
    end
    max_x = maximum(x, dims=dims)
    exp_x = exp.(x .- max_x)
    sum_exp = sum(exp_x, dims=dims)
    return exp_x ./ sum_exp
end

# Enhanced softmax supporting multiple types and dimensions
function softmax(x::AbstractArray{T}; dims=-1) where T<:Real
    return vectorized_softmax(x; dims=dims)
end

# Legacy softmax for backward compatibility
function softmax(x::Vector{T}) where T<:Real
    max_x = maximum(x)
    exp_x = exp.(x .- max_x)
    return exp_x ./ sum(exp_x)
end

# RMSD calculation with optimization
function rmsd(coords1::Vector{Tuple{Float64,Float64,Float64}},
              coords2::Vector{Tuple{Float64,Float64,Float64}})
    @assert length(coords1) == length(coords2) "Coordinate arrays must have same length"
    sum_sq_dist = sum(euclidean_distance(c1, c2)^2 for (c1, c2) in zip(coords1, coords2))
    return sqrt(sum_sq_dist / length(coords1))
end

# Optimized matrix multiplication with BLAS
function optimized_matmul(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T<:Union{Float32, Float64}
    if size(A, 2) != size(B, 1)
        error("Matrix dimensions don't match for multiplication: $(size(A)) × $(size(B))")
    end
    # Use BLAS for optimized matrix multiplication
    C = similar(A, size(A, 1), size(B, 2))
    return BLAS.gemm!('N', 'N', one(T), A, B, zero(T), C)
end

# Fallback for non-BLAS types
function optimized_matmul(A::AbstractMatrix, B::AbstractMatrix)
    return A * B
end

# Sparse matrix operations
function to_sparse_distance_matrix(dense_matrix::AbstractMatrix, threshold::Float32=1e-6)
    @debug "Converting to sparse matrix: $(size(dense_matrix)), threshold=$threshold"
    return sparse(dense_matrix .* (abs.(dense_matrix) .> threshold))
end

# Accelerated distance matrix with threading
function accelerated_distance_matrix(coords::AbstractMatrix, mask::Union{AbstractMatrix{Bool}, Nothing}=nothing, use_sparse::Bool=false)
    @debug "Computing optimized distance matrix with threading"
    n_atoms = size(coords, 1)
    result = zeros(Float32, n_atoms, n_atoms)

    @threads for i in 1:n_atoms
        for j in 1:n_atoms
            if exists(mask) && !mask[i, j]
                result[i, j] = 0.0f0
            else
                diff_x = coords[i, 1] - coords[j, 1]
                diff_y = coords[i, 2] - coords[j, 2]
                diff_z = coords[i, 3] - coords[j, 3]
                result[i, j] = sqrt(diff_x^2 + diff_y^2 + diff_z^2)
            end
        end
    end

    return use_sparse ? to_sparse_distance_matrix(result) : result
end

# Enhanced logging-safe operations
function log(t::AbstractArray, eps=1e-20)
    @debug "Computing log: shape=$(size(t)), type=$(eltype(t))"
    return log.(max.(t, eps))
end

# Utility functions
function divisible_by(num::Int, den::Int)
    @debug "Checking divisibility: $num by $den"
    return num % den == 0
end

function compact(args...)
    @debug "Compacting arguments: $(length(args)) items"
    return filter(exists, collect(args))
end

# Tensor operations
function l2norm(t::AbstractArray, dim=-1, eps=1e-20)
    @debug "L2 normalization: shape=$(size(t)), dim=$dim"
    if dim == -1
        dim = ndims(t)
    end
    norm = sqrt.(sum(t .^ 2, dims=dim) .+ eps)
    return t ./ norm
end

function max_neg_value(t::AbstractArray)
    @debug "Computing max negative value: type=$(eltype(t))"
    return -floatmax(eltype(t))
end

# Device management (CPU-only for Replit)
function to_device(x, dev_type::String="cpu")
    return x
end

function dict_to_device(d::Dict, device::String)
    return d
end

# Tensor manipulation
function symmetrize(t::AbstractArray{T,3}) where T
    @debug "Symmetrizing tensor: shape=$(size(t))"
    return t + permutedims(t, (2,1,3))
end

function masked_average(t::AbstractArray, mask::AbstractArray, dims, eps=1.0)
    @debug "Masked average: t_shape=$(size(t)), mask_shape=$(size(mask)), dims=$dims"
    num = sum(t .* mask, dims=dims)
    den = sum(mask, dims=dims)
    return num ./ max.(den, eps)
end

function exclusive_cumsum(t::AbstractArray, dim=-1)
    @debug "Exclusive cumulative sum: shape=$(size(t)), dim=$dim"
    if dim == -1
        dim = ndims(t)
    end
    s = size(t)
    cum = cumsum(t, dims=dim)
    shifted = zeros(eltype(t), s...)

    if dim == 1
        shifted[2:end, :] = cum[1:end-1, :]
    elseif dim == 2
        shifted[:, 2:end] = cum[:, 1:end-1]
    elseif dim == 3
        shifted[:, :, 2:end] = cum[:, :, 1:end-1]
    end
    return shifted
end

# Padding operations
function pad(t::AbstractArray, pads)
    @debug "Padding tensor: shape=$(size(t)), pads=$pads"
    padded = zeros(eltype(t), size(t) .+ pads...)
    slices = [1:size(t,i) for i in 1:ndims(t)]
    padded[slices...] = t
    return padded
end

function unpad(t::AbstractArray, pads)
    @debug "Unpadding tensor: shape=$(size(t)), pads=$pads"
    slices = [pads[i]+1:size(t,i) for i in 1:ndims(t)]
    return t[slices...]
end

# Attention mechanism support
function reshape_for_attention(t::AbstractArray, num_heads::Int, dim_head::Int)
    @debug "Reshaping for attention: shape=$(size(t)), heads=$num_heads, dim_head=$dim_head"
    batch, seq, dim = size(t)
    reshaped = reshape(t, batch, seq, num_heads, dim_head)
    return permutedims(reshaped, (1, 3, 2, 4))
end

end # module EnhancedSelfContainedMath
# =======================================================================


# =======================================================================
