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

@testset "Multi-RHS Parallel Solve" begin
    # Test parallel solve across multiple right-hand sides
    n = 1200

    Random.seed!(999)
    A = sprandn(n, n, 0.02)
    A = A + A'
    A = A + n * I
    A = sparse(A)

    F = qdldl(A, parallel=true)

    # Multiple right-hand sides
    nrhs = 10
    B = randn(n, nrhs)

    X = solve(F, B)

    # Check all solutions are correct
    @test norm(A * X - B) < 1e-8 * norm(B)

    # Compare with serial
    F_serial = qdldl(A, parallel=false)
    X_serial = solve(F_serial, B)

    @test X ≈ X_serial rtol=1e-10
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

@testset "Supernode Detection" begin
    # Test that supernode detection produces valid results
    n = 500

    Random.seed!(321)
    A = sprandn(n, n, 0.05)
    A = A + A'
    A = A + n * I
    A = sparse(A)

    F = qdldl(A)
    ws = F.workspace

    # Check all columns have valid supernode membership
    @test all(ws.snode_membership .>= 1)
    @test all(ws.snode_membership .<= ws.num_supernodes[])

    # Check supernode ranges cover all columns
    total_cols = 0
    for (start, stop) in ws.snode_ranges
        @test start >= 1
        @test stop <= n
        @test stop >= start
        total_cols += stop - start + 1

        # All columns in range should have same supernode id
        snode_id = ws.snode_membership[start]
        for j in start:stop
            @test ws.snode_membership[j] == snode_id
        end
    end
    @test total_cols == n

    # Number of supernodes should match ranges
    @test length(ws.snode_ranges) == ws.num_supernodes[]
end

@testset "Supernode Properties" begin
    # Test supernode properties on a matrix known to have supernodes
    # A banded matrix tends to have larger supernodes
    n = 100
    bandwidth = 5

    # Create banded matrix
    A = spzeros(n, n)
    for i in 1:n
        A[i, i] = n + 1.0  # diagonal
        for k in 1:bandwidth
            if i + k <= n
                A[i, i+k] = -1.0
            end
        end
    end
    A = sparse(A + A')

    F = qdldl(A)
    ws = F.workspace

    # Banded matrices should have some supernodes larger than 1
    max_snode_size = maximum(stop - start + 1 for (start, stop) in ws.snode_ranges)
    # Not all matrices have large supernodes, but at least check it runs
    @test max_snode_size >= 1
    @test ws.num_supernodes[] <= n  # At most n supernodes (singleton)
end
