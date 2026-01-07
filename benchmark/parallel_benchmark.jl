#!/usr/bin/env julia
# Benchmark comparing serial vs parallel QDLDL performance
#
# Run with: julia --threads=N --project benchmark/parallel_benchmark.jl
#
# Note: Currently parallelism benefits multi-RHS solves only.
# Factorization uses serial path due to race conditions in level-based approach.

using QDLDL, SparseArrays, LinearAlgebra, Random

function generate_test_matrix(n::Int, density::Float64, seed::Int=42)
    Random.seed!(seed)
    A = sprandn(n, n, density)
    A = A + A'  # Symmetric
    A = A + n * I  # Make positive definite
    return sparse(A)
end

function benchmark_factorization(A, ntrials=5)
    # Warmup
    _ = qdldl(A, parallel=false)
    _ = qdldl(A, parallel=true)

    times_serial = Float64[]
    times_parallel = Float64[]

    for _ in 1:ntrials
        t = @elapsed F_serial = qdldl(A, parallel=false)
        push!(times_serial, t)

        t = @elapsed F_parallel = qdldl(A, parallel=true)
        push!(times_parallel, t)
    end

    return minimum(times_serial), minimum(times_parallel)
end

function benchmark_solve_single_rhs(F_serial, F_parallel, b, ntrials=10)
    # Warmup
    _ = solve(F_serial, b)
    _ = solve(F_parallel, b)

    times_serial = Float64[]
    times_parallel = Float64[]

    for _ in 1:ntrials
        t = @elapsed _ = solve(F_serial, b)
        push!(times_serial, t)

        t = @elapsed _ = solve(F_parallel, b)
        push!(times_parallel, t)
    end

    return minimum(times_serial), minimum(times_parallel)
end

function benchmark_solve_multi_rhs(F_serial, F_parallel, B, ntrials=10)
    # Warmup
    _ = solve(F_serial, B)
    _ = solve(F_parallel, B)

    times_serial = Float64[]
    times_parallel = Float64[]

    for _ in 1:ntrials
        t = @elapsed _ = solve(F_serial, B)
        push!(times_serial, t)

        t = @elapsed _ = solve(F_parallel, B)
        push!(times_parallel, t)
    end

    return minimum(times_serial), minimum(times_parallel)
end

function run_benchmarks()
    println("=" ^ 70)
    println("QDLDL Parallel Benchmark")
    println("Julia threads: $(Threads.nthreads())")
    println("=" ^ 70)
    println()

    # Test sizes (keep small for reasonable runtime)
    sizes = [1000, 2000, 3000]
    density = 0.01
    nrhs_values = [1, 4, 8, 16, 32]

    println("Matrix generation: density = $density")
    println()

    # Factorization benchmark
    println("-" ^ 70)
    println("FACTORIZATION BENCHMARK")
    println("-" ^ 70)
    println("Note: Serial factorization used (parallel has too much sync overhead)")
    println()

    for n in sizes
        A = generate_test_matrix(n, density)
        nnz_A = nnz(A)

        t_serial, t_parallel = benchmark_factorization(A)

        println("n = $n, nnz = $nnz_A")
        println("  Serial:   $(round(t_serial * 1000, digits=2)) ms")
        println("  Parallel: $(round(t_parallel * 1000, digits=2)) ms")
        println("  Speedup:  $(round(t_serial / t_parallel, digits=2))x")
        println()
    end

    # Multi-RHS solve benchmark (where parallelism actually helps)
    println("-" ^ 70)
    println("MULTI-RHS SOLVE BENCHMARK (Parallelized)")
    println("-" ^ 70)
    println()

    n = 2000
    A = generate_test_matrix(n, density)

    F_serial = qdldl(A, parallel=false)
    F_parallel = qdldl(A, parallel=true)

    println("Matrix size: n = $n, nnz = $(nnz(A))")
    println()

    println("nrhs | Serial (ms) | Parallel (ms) | Speedup")
    println("-" ^ 50)

    for nrhs in nrhs_values
        Random.seed!(123)
        B = randn(n, nrhs)

        t_serial, t_parallel = benchmark_solve_multi_rhs(F_serial, F_parallel, B)

        serial_ms = round(t_serial * 1000, digits=2)
        parallel_ms = round(t_parallel * 1000, digits=2)
        speedup = round(t_serial / t_parallel, digits=2)

        println("$(lpad(nrhs, 4)) | $(lpad(serial_ms, 11)) | $(lpad(parallel_ms, 13)) | $(lpad(speedup, 7))x")
    end

    println()
    println("=" ^ 70)
    println("Benchmark complete")
    println("=" ^ 70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
