using QDLDL, SparseArrays, LinearAlgebra, Test, Random

@testset "Parallel Factorization" begin

    # Test that parallel produces same results as serial
    # Use a matrix larger than PARALLEL_THRESHOLD for parallel path
    n = 1500  # > PARALLEL_THRESHOLD (1000)

    # Create a random positive definite sparse matrix
    Random.seed!(42)
    A = sprandn(n, n, 0.01)
    A = A + A'  # Symmetric
    A = A + n * I  # Make strongly diagonally dominant (positive definite)
    A = sparse(A)

    # Factor with serial and parallel
    F_serial = qdldl(A, parallel=false)
    F_parallel = qdldl(A, parallel=true)

    # Compare factorizations
    @test F_serial.workspace.D ≈ F_parallel.workspace.D rtol=1e-10
    @test F_serial.workspace.Dinv ≈ F_parallel.workspace.Dinv rtol=1e-10

    # The L matrix values should be approximately equal
    # Note: order of entries within columns may differ in parallel version
    L_serial = F_serial.L
    L_parallel = F_parallel.L
    @test norm(L_serial - L_parallel) < 1e-10 * norm(L_serial)

    # Test solve produces same results
    b = randn(n)
    x_serial = solve(F_serial, b)
    x_parallel = solve(F_parallel, b)

    @test x_serial ≈ x_parallel rtol=1e-10

    # Verify solution is correct
    @test A * x_serial ≈ b rtol=1e-8
    @test A * x_parallel ≈ b rtol=1e-8

    # Test positive inertia matches
    @test positive_inertia(F_serial) == positive_inertia(F_parallel)
end

@testset "Parallel with Small Matrix (Falls back to Serial)" begin
    # Small matrix should use serial even with parallel=true
    n = 100  # < PARALLEL_THRESHOLD

    Random.seed!(123)
    A = sprandn(n, n, 0.1)
    A = A + A'  # Symmetric
    A = A + n * I
    A = sparse(A)

    F = qdldl(A, parallel=true)

    # Should not actually use parallel (matrix too small)
    @test F.parallel[] == false || Threads.nthreads() == 1

    # But should still produce correct result
    b = randn(n)
    x = solve(F, b)
    @test A * x ≈ b rtol=1e-8
end

@testset "Parallel Solve Correctness" begin
    # Test multiple solves with same factorization
    n = 1200

    Random.seed!(456)
    A = sprandn(n, n, 0.02)
    A = A + A'  # Symmetric
    A = A + n * I
    A = sparse(A)

    F = qdldl(A, parallel=true)

    # Multiple right-hand sides
    for _ in 1:5
        b = randn(n)
        x = solve(F, b)
        @test A * x ≈ b rtol=1e-8
    end
end

@testset "Level Computation" begin
    # Test that level computation produces valid levels
    n = 500

    Random.seed!(789)
    A = sprandn(n, n, 0.05)
    A = A + A'  # Symmetric
    A = A + n * I
    A = sparse(A)

    F = qdldl(A, parallel=true)
    ws = F.workspace

    # Check all nodes have valid levels
    @test all(ws.levels .>= 0)
    @test ws.max_level[] >= 0

    # Check level_sets cover all nodes
    all_nodes = reduce(vcat, ws.level_sets)
    @test sort(all_nodes) == collect(1:n)

    # Check level consistency with elimination tree
    for i in 1:n
        level_i = ws.levels[i]
        parent = ws.etree[i]
        if parent != -1 && parent <= n  # Not root
            @test ws.levels[parent] > level_i  # Parent at higher level
        end
    end
end
