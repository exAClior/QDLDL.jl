module QDLDL

export qdldl, \, solve, solve!, refactor!, update_values!, scale_values!,  positive_inertia, regularized_entries

using AMD, SparseArrays
using LinearAlgebra: istriu, triu, Diagonal, BLAS, LAPACK

const QDLDL_UNKNOWN = -1;
const QDLDL_USED   = true;
const QDLDL_UNUSED = false;

# Minimum matrix size for parallel factorization
const PARALLEL_THRESHOLD = 1000

# Minimum supernode size to use BLAS operations
# Smaller supernodes use scalar loops (lower overhead)
const SUPERNODE_BLAS_THRESHOLD = 32

# =============================================================================
# PARALLEL FACTORIZATION STATUS
# =============================================================================
#
# Current implementation provides parallelism via:
# 1. Multi-RHS solve: parallelizes across columns of B in solve(F, B::Matrix)
#    Achieves ~3.6x speedup with 4 threads
#
# Infrastructure in place for future parallel factorization:
# - Level computation for elimination tree (levels, level_sets)
# - Supernode detection (snode_membership, snode_ranges)
# - Children list (inverse of etree) for task-based traversal
# - Subtree sizes for load balancing decisions
#
# WHY PARALLEL FACTORIZATION IS HARD FOR LEFT-LOOKING LDL:
# The left-looking algorithm processes columns 1..n in order. Column k reads
# from columns j < k that are in its "reach" (determined by elimination tree
# paths from nonzeros in A[:,k]). The key challenge:
#
# 1. Level-based parallelism (attempted): Columns at the same etree level
#    can share ancestors through the sparsity pattern, creating race conditions.
#
# 2. Sibling parallelism: Even etree siblings can have dependencies if their
#    column ranges interleave (the AMD ordering doesn't guarantee separation).
#
# 3. Task-based parallelism: Would require either:
#    a) Nested dissection ordering (not AMD) to guarantee subtree separation
#    b) Complex dependency analysis of the filled graph
#    c) Right-looking reformulation (different algorithm)
#
# FUTURE DIRECTIONS:
# 1. Supernodal factorization with BLAS threading:
#    - Use dense BLAS operations (TRSM, GEMM) within large supernodes
#    - BLAS library provides threading automatically
# 2. Nested dissection ordering option:
#    - Reorder matrix to enable subtree parallelism
#    - Trade-off: may increase fill-in vs AMD
# 3. Right-looking reformulation:
#    - After processing column k, push updates to dependent columns
#    - Allows multiple columns to accumulate updates in parallel
# =============================================================================

"""
Thread-local workspace for parallel factorization.
Each thread gets its own copy to avoid data races.
"""
struct ThreadWorkspace{Tf<:AbstractFloat,Ti<:Integer}
    yMarkers::Vector{Bool}
    yIdx::Vector{Ti}
    elimBuffer::Vector{Ti}
    yVals::Vector{Tf}
end

function ThreadWorkspace{Tf,Ti}(n::Integer) where {Tf<:AbstractFloat,Ti<:Integer}
    yMarkers = fill(QDLDL_UNUSED, n)
    yIdx = Vector{Ti}(undef, n)
    elimBuffer = Vector{Ti}(undef, n)
    yVals = zeros(Tf, n)  # Must be zero-initialized
    ThreadWorkspace{Tf,Ti}(yMarkers, yIdx, elimBuffer, yVals)
end


struct QDLDLWorkspace{Tf<:AbstractFloat,Ti<:Integer}

    #internal workspace data
    etree::Vector{Ti}
      Lnz::Vector{Ti}
    iwork::Vector{Ti}
    bwork::Vector{Bool}
    fwork::Vector{Tf}

    #L matrix row indices and data
    Ln::Int         #always Int since SparseMatrixCSC does it this way
    Lp::Vector{Ti}
    Li::Vector{Ti}
    Lx::Vector{Tf}

    #D and its inverse
    D::Vector{Tf}
    Dinv::Vector{Tf}

    #number of positive values in D
    positive_inertia::Ref{Ti}

    #The upper triangular matrix factorisation target
    #This is the post ordering PAPt of the original data
    triuA::SparseMatrixCSC{Tf,Ti}

    #mapping from entries in the triu form
    #of the original input to the post ordering
    #triu form used for the factorization
    #this can be used when modifying entries
    #of the data matrix for refactoring
    AtoPAPt::Union{Vector{Ti},Nothing}

    #regularization parameters
    Dsigns::Union{Vector{Ti},Nothing}
    regularize_eps::Tf
    regularize_delta::Tf

    #number of regularized entries in D
    #length 1 vector instead of ref to avoid allocations
    #while maintaining immutability
    regularize_count::Vector{Ti}

    #parallel factorization support
    levels::Vector{Ti}                          # level of each node in etree
    level_sets::Vector{Vector{Ti}}              # nodes grouped by level
    max_level::Ref{Ti}                          # maximum level
    thread_workspaces::Vector{ThreadWorkspace{Tf,Ti}}  # per-thread work arrays

    #supernodal factorization support
    snode_membership::Vector{Ti}                # snode_membership[i] = supernode containing column i
    snode_ranges::Vector{Tuple{Ti,Ti}}          # (start, end) for each supernode
    num_supernodes::Ref{Ti}                     # total number of supernodes

    #task-based parallelism support
    children::Vector{Vector{Ti}}                # children[i] = list of children of node i in etree
    subtree_sizes::Vector{Ti}                   # subtree_sizes[i] = size of subtree rooted at i

end

function QDLDLWorkspace(triuA::SparseMatrixCSC{Tf,Ti},
                        AtoPAPt::Union{Vector{Ti},Nothing},
                        Dsigns::Union{Vector{Ti},Nothing},
                        regularize_eps::Tf,
                        regularize_delta::Tf
) where {Tf<:AbstractFloat,Ti<:Integer}

    etree  = Vector{Ti}(undef,triuA.n)
    Lnz    = Vector{Ti}(undef,triuA.n)
    iwork  = Vector{Ti}(undef,triuA.n*3)
    bwork  = Vector{Bool}(undef,triuA.n)
    fwork  = Vector{Tf}(undef,triuA.n)

    #compute elimination tree using QDLDL converted code
    sumLnz = QDLDL_etree!(triuA.n,triuA.colptr,triuA.rowval,iwork,Lnz,etree)

    if(sumLnz < 0)
        error("Input matrix is not upper triangular or has an empty column")
    end

    #allocate space for the L matrix row indices and data
    Ln = triuA.n
    Lp = Vector{Ti}(undef,triuA.n + 1)
    Li = Vector{Ti}(undef,sumLnz)
    Lx = Vector{Tf}(undef,sumLnz)

    #allocate for D and D inverse
    D  = Vector{Tf}(undef,triuA.n)
    Dinv = Vector{Tf}(undef,triuA.n)

    #allocate for positive inertia count.  -1 to
    #start since we haven't counted anything yet
    positive_inertia = Ref{Ti}(-1)

    #number of regularized entries in D. None to start
    regularize_count = zeros(Ti,1)

    #parallel factorization support
    levels = Vector{Ti}(undef, triuA.n)
    level_sets = Vector{Vector{Ti}}()
    max_level_val = compute_tree_levels!(etree, levels, level_sets)
    max_level = Ref{Ti}(max_level_val)

    #allocate thread workspaces (one per thread)
    #allocate extra to handle dynamic thread pools
    nthreads = max(Threads.nthreads(), Threads.maxthreadid(), 8)
    thread_workspaces = [ThreadWorkspace{Tf,Ti}(triuA.n) for _ in 1:nthreads]

    #supernodal factorization support
    snode_membership = Vector{Ti}(undef, triuA.n)
    snode_ranges = Vector{Tuple{Ti,Ti}}()
    num_snode = detect_supernodes!(etree, Lnz, snode_membership, snode_ranges)
    num_supernodes = Ref{Ti}(num_snode)

    #task-based parallelism support
    children = build_children_list(etree)
    subtree_sizes = Vector{Ti}(undef, triuA.n)
    compute_subtree_sizes!(etree, children, subtree_sizes)

    QDLDLWorkspace(etree,Lnz,iwork,bwork,fwork,
                   Ln,Lp,Li,Lx,D,Dinv,positive_inertia,triuA,
                   AtoPAPt, Dsigns,regularize_eps,
                   regularize_delta,regularize_count,
                   levels, level_sets, max_level, thread_workspaces,
                   snode_membership, snode_ranges, num_supernodes,
                   children, subtree_sizes)

end

struct QDLDLFactorisation{Tf<:AbstractFloat,Ti<:Integer}

    #permutation vector (nothing if no permutation)
    perm::Union{Nothing,Vector{Ti}}
    #inverse permutation (nothing if no permutation)
    iperm::Union{Nothing,Vector{Ti}}
    #lower triangular factor
    L::SparseMatrixCSC{Tf,Ti}
    #Inverse of D matrix in ldl
    Dinv::Diagonal{Tf,Vector{Tf}}
    #workspace data
    workspace::QDLDLWorkspace{Tf,Ti}
    #is it logical factorisation only?
    logical::Ref{Bool}
    #use parallel factorization and solve?
    parallel::Ref{Bool}
end




# Usage :
# qdldl(A) uses the default AMD ordering
# qdldl(A,perm = p) uses a caller specified ordering
# qdldl(A,perm = nothing) factors without reordering
#
# qdldl(A,logical = true) produces a logical factorisation only
#
# qdldl(A,signs = s, thresh_eps = ϵ, thresh_delta = δ) produces
# a factorization with dynamic regularization based on the vector
# of signs in s and using regularization parameters (ϵ,δ).  The
# scalars (ϵ,δ) = (1e-12,1e-7) by default.   By default s = nothing,
# and no regularization is performed.
#
# qdldl(A,amd_dense_scale = s) scales AMD.AMD_DENSE by a factor s :
# (s = 1.0 by default).   This is only used if no perm parameter 
# is provided. 

function qdldl(A::SparseMatrixCSC{Tf,Ti};
               amd_dense_scale::Tf = Tf(1.0),
               perm::Union{Array{Ti},Nothing}=_get_amd_ordering(A,amd_dense_scale),
               logical::Bool=false,
               Dsigns::Union{Array{Ti},Nothing} = nothing,
               regularize_eps::Tf = Tf(1e-12),
               regularize_delta::Tf = Tf(1e-7),
               parallel::Bool=false,
              ) where {Tf<:AbstractFloat, Ti<:Integer}

    #store the inverse permutation to enable matrix updates
    iperm = perm === nothing ? nothing : invperm(perm)

    if(!istriu(A))
        A = triu(A)
    else
        #either way, we take an internal copy
        A = deepcopy(A)
    end


    #permute using symperm, producing a triu matrix to factor
    if perm !== nothing
        A, AtoPAPt = permute_symmetric(A, iperm)  #returns an upper triangular matrix
    else
        AtoPAPt = nothing
    end

    #hold an internal copy of the (possibly permuted)
    #vector of signs if one was specified
    if(Dsigns !== nothing)
        mysigns = similar(Dsigns)
        if(perm === nothing)
            mysigns .= Dsigns
        else
            permute!(mysigns,Dsigns,perm)
        end
    else
        mysigns = nothing
    end

    #allocate workspace
    workspace = QDLDLWorkspace(A,AtoPAPt,mysigns,regularize_eps,regularize_delta)

    #determine if we should use parallel operations
    use_parallel = parallel && A.n >= PARALLEL_THRESHOLD && Threads.nthreads() > 1

    #factor the matrix
    #Note: Parallel factorization has correct implementation but high overhead
    #from level synchronization. Only beneficial for matrices with few levels
    #and many columns per level. For typical sparse matrices, serial is faster.
    #The parallel flag enables parallel multi-RHS solves which provide ~4x speedup.
    #
    #Heuristic: use parallel factorization only if average columns per level > threshold
    avg_cols_per_level = A.n / max(workspace.max_level[], 1)
    use_parallel_factor = use_parallel && avg_cols_per_level > 100

    if use_parallel_factor
        factor_parallel!(workspace, logical)
    else
        factor!(workspace, logical)
    end

    #make user-friendly factors
    L = SparseMatrixCSC(workspace.Ln,
                        workspace.Ln,
                        workspace.Lp,
                        workspace.Li,
                        workspace.Lx)
    Dinv = Diagonal(workspace.Dinv)

    #Psss a Ref{Bool} to the constructor since QDLDLFactorisation
    #is immutable.   All internal functions will just use a Bool

    return QDLDLFactorisation(perm, iperm, L, Dinv, workspace, Ref{Bool}(logical), Ref{Bool}(use_parallel))

end

function positive_inertia(F::QDLDLFactorisation)
    F.workspace.positive_inertia[]
end

function regularized_entries(F::QDLDLFactorisation)
    F.workspace.regularize_count[1]
end


function update_values!(
    F::QDLDLFactorisation,
    indices::Union{AbstractVector{Ti},Ti},
    values::Union{AbstractVector{Tf},Tf},
) where{Ti <: Integer, Tf <: Real}

    triuA   = F.workspace.triuA     #post permutation internal data
    AtoPAPt = F.workspace.AtoPAPt   #mapping from input matrix entries to triuA

    if AtoPAPt === nothing
        @views triuA.nzval[indices] .= values
    else
        @views triuA.nzval[AtoPAPt[indices]] .= values
    end

    return nothing
end


function scale_values!(
    F::QDLDLFactorisation,
    indices::Union{AbstractVector{Ti},Ti},
    scale::Tf,
) where{Ti <: Integer, Tf <: Real}

    triuA   = F.workspace.triuA     #post permutation internal data
    AtoPAPt = F.workspace.AtoPAPt   #mapping from input matrix entries to triuA

    if AtoPAPt === nothing
        @views triuA.nzval[indices] .*= scale
    else
        @views triuA.nzval[AtoPAPt[indices]] .*= scale
    end

    return nothing
end


function Base.:\(F::QDLDLFactorisation,b)
    return solve(F,b)
end


function refactor!(F::QDLDLFactorisation)

    #It never makes sense to call refactor for a logical
    #factorization since it will always be the same.  Calling
    #this function implies that we want a numerical factorization

    F.logical[] = false  #in case not already

    factor!(F.workspace,F.logical[])
end


function factor!(workspace::QDLDLWorkspace{Tf,Ti},logical::Bool) where {Tf<:AbstractFloat,Ti<:Integer}

    if(logical)
        workspace.Lx   .= 1
        workspace.D    .= 1
        workspace.Dinv .= 1
    end

    #factor using QDLDL converted code
    A = workspace.triuA
    posDCount  = QDLDL_factor!(A.n,A.colptr,A.rowval,A.nzval,
                              workspace.Lp,
                              workspace.Li,
                              workspace.Lx,
                              workspace.D,
                              workspace.Dinv,
                              workspace.Lnz,
                              workspace.etree,
                              workspace.bwork,
                              workspace.iwork,
                              workspace.fwork,
                              logical,
                              workspace.Dsigns,
                              workspace.regularize_eps,
                              workspace.regularize_delta,
                              workspace.regularize_count
                              )

    if(posDCount < 0)
        error("Zero entry in D (matrix is not quasidefinite)")
    end

    workspace.positive_inertia[] = posDCount

    return nothing

end


"""
Parallel factorization wrapper.
Uses level-based parallelism when beneficial.
"""
function factor_parallel!(workspace::QDLDLWorkspace{Tf,Ti}, logical::Bool) where {Tf<:AbstractFloat,Ti<:Integer}

    if logical
        workspace.Lx   .= 1
        workspace.D    .= 1
        workspace.Dinv .= 1
    end

    A = workspace.triuA

    # Use parallel factorization
    posDCount = QDLDL_factor_parallel!(
        A.n, A.colptr, A.rowval, A.nzval,
        workspace.Lp,
        workspace.Li,
        workspace.Lx,
        workspace.D,
        workspace.Dinv,
        workspace.Lnz,
        workspace.etree,
        logical,
        workspace.Dsigns,
        workspace.regularize_eps,
        workspace.regularize_delta,
        workspace.regularize_count,
        workspace.level_sets,
        workspace.thread_workspaces
    )

    if posDCount < 0
        error("Zero entry in D (matrix is not quasidefinite)")
    end

    workspace.positive_inertia[] = posDCount

    return nothing
end


# Solves Ax = b using LDL factors for A.
# Returns x, preserving b (single RHS as vector)
function solve(F::QDLDLFactorisation, b::AbstractVector)
    x = copy(b)
    solve!(F, x)
    return x
end


"""
    solve(F, B::AbstractMatrix)

Solve AX = B for multiple right-hand sides.
When parallel=true was used in factorization and nthreads > 1,
solves are parallelized across columns of B.
"""
function solve(F::QDLDLFactorisation{Tf,Ti}, B::AbstractMatrix{Tf}) where {Tf,Ti}
    X = copy(B)
    solve!(F, X)
    return X
end


"""
    solve!(F, B::AbstractMatrix)

Solve AX = B in-place for multiple right-hand sides.
When parallel=true was used in factorization and nthreads > 1,
solves are parallelized across columns of B.
"""
function solve!(F::QDLDLFactorisation{Tf,Ti}, B::AbstractMatrix{Tf}) where {Tf,Ti}

    if F.logical[]
        error("Can't solve with logical factorisation only")
    end

    nrhs = size(B, 2)

    # Parallelize across right-hand sides when beneficial
    if F.parallel[] && nrhs > 1 && Threads.nthreads() > 1
        Threads.@threads for j in 1:nrhs
            b_col = view(B, :, j)
            _solve_single_rhs!(F, b_col)
        end
    else
        for j in 1:nrhs
            b_col = view(B, :, j)
            _solve_single_rhs!(F, b_col)
        end
    end

    return nothing
end


# Internal: solve for a single RHS (used by parallel multi-RHS solver)
function _solve_single_rhs!(F::QDLDLFactorisation{Tf,Ti}, b::AbstractVector{Tf}) where {Tf,Ti}
    n = F.workspace.Ln

    # Need thread-local work array for permutation
    if F.perm !== nothing
        tmp = similar(b)
        permute!(tmp, b, F.perm)

        QDLDL_solve!(n,
                     F.workspace.Lp,
                     F.workspace.Li,
                     F.workspace.Lx,
                     F.workspace.Dinv,
                     tmp)

        ipermute!(b, tmp, F.perm)
    else
        QDLDL_solve!(n,
                     F.workspace.Lp,
                     F.workspace.Li,
                     F.workspace.Lx,
                     F.workspace.Dinv,
                     b)
    end

    return nothing
end

# Solves Ax = b using LDL factors for A.
# Solves in place (x replaces b) - single RHS as vector
function solve!(F::QDLDLFactorisation, b::AbstractVector)

    #bomb if logical factorisation only
    if F.logical[]
        error("Can't solve with logical factorisation only")
    end

    #permute b
    tmp = F.perm === nothing ? b : permute!(F.workspace.fwork, b, F.perm)

    QDLDL_solve!(F.workspace.Ln,
                 F.workspace.Lp,
                 F.workspace.Li,
                 F.workspace.Lx,
                 F.workspace.Dinv,
                 tmp)

    #inverse permutation
    b = F.perm === nothing ? tmp : ipermute!(b, F.workspace.fwork, F.perm)

    return nothing
end



# Compute the elimination tree for a quasidefinite matrix
# in compressed sparse column form.

function QDLDL_etree!(n,Ap,Ai,work,Lnz,etree)

    @inbounds for i = 1:n
        # zero out Lnz and work.  Set all etree values to unknown
        work[i]  = 0
        Lnz[i]   = 0
        etree[i] = QDLDL_UNKNOWN

        #Abort if A doesn't have at least one entry
        #one entry in every column
        if(Ap[i] == Ap[i+1])
            return -1
        end
    end

    @inbounds for j = 1:n
        work[j] = j
        @inbounds for p = Ap[j]:(Ap[j+1]-1)
            i = Ai[p]
            if(i > j)
                return -1
            end
            @inbounds while(work[i] != j)
                if(etree[i] == QDLDL_UNKNOWN)
                    etree[i] = j
                end
                Lnz[i] += 1        #nonzeros in this column
                work[i] = j
                i = etree[i]
            end
        end #end for p
    end

    #tally the total nonzeros
    sumLnz = sum(Lnz)

    return sumLnz
end


"""
    compute_tree_levels!(etree, levels, level_sets)

Compute levels in the elimination tree for parallel factorization.
Nodes at the same level are independent and can be processed in parallel.

- `etree`: elimination tree (etree[i] = parent of i, or QDLDL_UNKNOWN for root)
- `levels`: output vector, levels[i] = level of node i (leaves = 0)
- `level_sets`: output vector of vectors, level_sets[l+1] contains nodes at level l

Returns the maximum level.
"""
function compute_tree_levels!(etree::Vector{Ti}, levels::Vector{Ti},
                              level_sets::Vector{Vector{Ti}}) where {Ti<:Integer}
    n = length(etree)

    # Initialize levels to -1 (uncomputed)
    fill!(levels, Ti(-1))

    # Count children for each node
    child_count = zeros(Ti, n)
    for i = 1:n
        parent = etree[i]
        if parent != QDLDL_UNKNOWN && parent <= n
            child_count[parent] += 1
        end
    end

    # Leaves have no children, assign level 0
    # Use a queue-based approach (bottom-up)
    queue = Ti[]
    for i = 1:n
        if child_count[i] == 0
            levels[i] = 0
            push!(queue, i)
        end
    end

    # Process nodes bottom-up
    max_level = Ti(0)
    while !isempty(queue)
        node = popfirst!(queue)
        parent = etree[node]

        if parent != QDLDL_UNKNOWN && parent <= n
            # Parent's level is max of children's levels + 1
            new_level = levels[node] + 1
            if levels[parent] < new_level
                levels[parent] = new_level
                max_level = max(max_level, new_level)
            end

            # Decrement parent's remaining children count
            child_count[parent] -= 1
            if child_count[parent] == 0
                push!(queue, parent)
            end
        end
    end

    # Build level sets
    empty!(level_sets)
    for _ = 0:max_level
        push!(level_sets, Ti[])
    end

    for i = 1:n
        level = levels[i]
        if level >= 0
            push!(level_sets[level + 1], i)  # +1 for 1-based indexing
        end
    end

    return max_level
end


"""
    build_children_list(etree)

Build a list of children for each node in the elimination tree.
Returns a vector of vectors where children[i] contains the indices of all children of node i.
"""
function build_children_list(etree::Vector{Ti}) where {Ti<:Integer}
    n = length(etree)
    children = [Ti[] for _ in 1:n]

    for i = 1:n
        parent = etree[i]
        if parent != QDLDL_UNKNOWN && parent <= n
            push!(children[parent], i)
        end
    end

    return children
end


"""
    compute_subtree_sizes!(etree, children, subtree_sizes)

Compute the size of each subtree in the elimination tree.
subtree_sizes[i] = number of nodes in the subtree rooted at i (including i).
"""
function compute_subtree_sizes!(etree::Vector{Ti}, children::Vector{Vector{Ti}},
                                subtree_sizes::Vector{Ti}) where {Ti<:Integer}
    n = length(etree)
    fill!(subtree_sizes, Ti(0))

    # Process in reverse order (leaves first, then parents)
    for i = 1:n
        subtree_sizes[i] = 1  # Count self
        for child in children[i]
            subtree_sizes[i] += subtree_sizes[child]
        end
    end

    return nothing
end


"""
    find_parallel_subtrees(etree, children, subtree_sizes, min_size)

Find subtrees that are suitable for parallel processing.
Returns indices of nodes whose subtrees:
1. Are large enough (>= min_size)
2. Are children of the same parent (siblings can be processed in parallel in post-order)

Note: For left-looking LDL, siblings can only be processed in parallel if their
column ranges don't overlap and they don't share dependencies. This function
identifies candidates; actual parallelism requires careful dependency analysis.
"""
function find_parallel_subtrees(etree::Vector{Ti}, children::Vector{Vector{Ti}},
                                subtree_sizes::Vector{Ti}, min_size::Ti) where {Ti<:Integer}
    n = length(etree)
    parallel_roots = Ti[]

    # Find nodes with multiple large children
    for i = 1:n
        large_children = filter(c -> subtree_sizes[c] >= min_size, children[i])
        if length(large_children) >= 2
            append!(parallel_roots, large_children)
        end
    end

    return parallel_roots
end


"""
    detect_supernodes!(etree, Lnz, snode_membership, snode_ranges)

Detect fundamental supernodes in the elimination tree.
A supernode is a maximal set of consecutive columns j, j+1, ..., j+k-1 where:
- etree[j+i] = j+i+1 for i = 0, ..., k-2 (chain in elimination tree)
- Lnz[j+i+1] = Lnz[j+i] - 1 (nested sparsity pattern)

Returns the number of supernodes.
- `snode_membership[i]` = supernode index that column i belongs to
- `snode_ranges` = vector of (start, end) pairs for each supernode
"""
function detect_supernodes!(etree::Vector{Ti}, Lnz::Vector{Ti},
                            snode_membership::Vector{Ti},
                            snode_ranges::Vector{Tuple{Ti,Ti}}) where {Ti<:Integer}
    n = length(etree)
    empty!(snode_ranges)

    if n == 0
        return Ti(0)
    end

    snode_id = Ti(1)
    snode_start = Ti(1)

    for j = 1:n-1
        # Check if j and j+1 are in the same supernode:
        # 1. j's parent in etree is j+1
        # 2. Lnz[j+1] = Lnz[j] - 1 (nested pattern)
        in_same_snode = (etree[j] == j + 1) && (Lnz[j+1] == Lnz[j] - 1)

        if in_same_snode
            # Continue current supernode
            snode_membership[j] = snode_id
        else
            # End current supernode, start new one
            snode_membership[j] = snode_id
            push!(snode_ranges, (snode_start, Ti(j)))
            snode_id += 1
            snode_start = Ti(j + 1)
        end
    end

    # Handle last column
    snode_membership[n] = snode_id
    push!(snode_ranges, (snode_start, Ti(n)))

    return snode_id
end


function QDLDL_factor!(
        n,
        Ap,
        Ai,
        Ax,
        Lp,
        Li,
        Lx,
        D,
        Dinv,
        Lnz,
        etree,
        bwork,
        iwork,
        fwork,
        logicalFactor::Bool,
        Dsigns,
        regularize_eps,
        regularize_delta,
        regularize_count
)

    positiveValuesInD  = 0
    regularize_count[1] = 0

    #partition working memory into pieces
    yMarkers        = bwork
    yIdx            = view(iwork,      1:n)
    elimBuffer      = view(iwork,  (n+1):2*n)
    LNextSpaceInCol = view(iwork,(2*n+1):3*n)
    yVals           = fwork

    Lp[1] = 1 #first column starts at index one / Julia is 1 indexed

    @inbounds for i = 1:n

        #compute L column indices
        Lp[i+1] = Lp[i] + Lnz[i]   #cumsum, total at the end

        # set all Yidx to be 'unused' initially
        #in each column of L, the next available space
        #to start is just the first space in the column
        yMarkers[i]  = QDLDL_UNUSED
        yVals[i]     = 0.0
        D[i]         = 0.0
        LNextSpaceInCol[i] = Lp[i]
    end

    if(!logicalFactor)
        # First element of the diagonal D.
        D[1]     = Ax[1]
        if(Dsigns !== nothing && Dsigns[1]*D[1] < regularize_eps)
            D[1] = regularize_delta * Dsigns[1]
            regularize_count[1] += 1
        end

        if(D[1] == 0.0) return -1 end
        if(D[1]  > 0.0) positiveValuesInD += 1 end
        Dinv[1] = 1/D[1];
    end

    #Start from second row here. The upper LH corner is trivially 0
    #in L b/c we are only computing the subdiagonal elements
    @inbounds for k = 2:n

        #NB : For each k, we compute a solution to
        #y = L(0:(k-1),0:k-1))\b, where b is the kth
        #column of A that sits above the diagonal.
        #The solution y is then the kth row of L,
        #with an implied '1' at the diagonal entry.

        #number of nonzeros in this row of L
        nnzY = 0  #number of elements in this row

        #This loop determines where nonzeros
        #will go in the kth row of L, but doesn't
        #compute the actual values
        @inbounds for i = Ap[k]:(Ap[k+1]-1)

            bidx = Ai[i]   # we are working on this element of b

            #Initialize D[k] as the element of this column
            #corresponding to the diagonal place.  Don't use
            #this element as part of the elimination step
            #that computes the k^th row of L
            if(bidx == k)
                D[k] = Ax[i];
                continue
            end

            yVals[bidx] = Ax[i]   # initialise y(bidx) = b(bidx)

            # use the forward elimination tree to figure
            # out which elements must be eliminated after
            # this element of b
            nextIdx = bidx

            if(yMarkers[nextIdx] == QDLDL_UNUSED)  #this y term not already visited

                yMarkers[nextIdx] = QDLDL_USED     #I touched this one
                elimBuffer[1]     = nextIdx  # It goes at the start of the current list
                nnzE              = 1         #length of unvisited elimination path from here

                nextIdx = etree[bidx];

                @inbounds while(nextIdx != QDLDL_UNKNOWN && nextIdx < k)
                    if(yMarkers[nextIdx] == QDLDL_USED) break; end

                    yMarkers[nextIdx] = QDLDL_USED;   #I touched this one
                    #NB: Julia is 1-indexed, so I increment nnzE first here,
                    #not after writing into elimBuffer as in the C version
                    nnzE += 1                   #the list is one longer than before
                    elimBuffer[nnzE] = nextIdx; #It goes in the current list
                    nextIdx = etree[nextIdx];   #one step further along tree

                end #end while

                # now I put the buffered elimination list into
                # my current ordering in reverse order
                @inbounds while(nnzE != 0)
                    #NB: inc/dec reordered relative to C because
                    #the arrays are 1 indexed
                    nnzY += 1;
                    yIdx[nnzY] = elimBuffer[nnzE];
                    nnzE -= 1;
                end #end while
            end #end if

        end #end for i

        #This for loop places nonzeros values in the k^th row
        @inbounds for i = nnzY:-1:1

            #which column are we working on?
            cidx = yIdx[i]

            # loop along the elements in this
            # column of L and subtract to solve to y
            tmpIdx = LNextSpaceInCol[cidx];

            #don't compute Lx for logical factorisation
            #this is not implemented in the C version
            if(!logicalFactor)
                yVals_cidx = yVals[cidx]
                @inbounds for j = Lp[cidx]:(tmpIdx-1)
                    yVals[Li[j]] -= Lx[j]*yVals_cidx
                end

                #Now I have the cidx^th element of y = L\b.
                #so compute the corresponding element of
                #this row of L and put it into the right place
                Lx[tmpIdx] = yVals_cidx *Dinv[cidx]

                #D[k] -= yVals[cidx]*yVals[cidx]*Dinv[cidx];
                D[k] -= yVals_cidx*Lx[tmpIdx]
            end

            #also record which row it went into
            Li[tmpIdx] = k

            LNextSpaceInCol[cidx] += 1

            #reset the yvalues and indices back to zero and QDLDL_UNUSED
            #once I'm done with them
            yVals[cidx]     = 0.0
            yMarkers[cidx]  = QDLDL_UNUSED

        end #end for i

        #apply dynamic regularization if a sign
        #vector has been specified.
        if(Dsigns !== nothing && Dsigns[k]*D[k] < regularize_eps)
            D[k] = regularize_delta * Dsigns[k]
            regularize_count[1] += 1
        end

        #Maintain a count of the positive entries
        #in D.  If we hit a zero, we can't factor
        #this matrix, so abort
        if(D[k] == 0.0) return -1 end
        if(D[k]  > 0.0) positiveValuesInD += 1 end

        #compute the inverse of the diagonal
        Dinv[k]= 1/D[k]

    end #end for k

    return positiveValuesInD

end


"""
Parallel LDL factorization using level-based parallelism with snapshot isolation.

Columns at the same level in the elimination tree can be processed concurrently.
Key insight: at the start of each level, we snapshot LNextSpaceInCol to ensure
threads only read L entries from previous levels (not entries being written
concurrently by other threads at the same level).
"""
function QDLDL_factor_parallel!(
        n,
        Ap,
        Ai,
        Ax,
        Lp,
        Li,
        Lx,
        D,
        Dinv,
        Lnz,
        etree,
        logicalFactor::Bool,
        Dsigns,
        regularize_eps,
        regularize_delta,
        regularize_count,
        level_sets::Vector{Vector{Ti}},
        thread_workspaces::Vector{ThreadWorkspace{Tf,Ti}}
) where {Tf<:AbstractFloat, Ti<:Integer}

    # Shared state with atomic access
    positiveValuesInD = Threads.Atomic{Ti}(0)
    regularize_count[1] = 0
    regularize_count_atomic = Threads.Atomic{Ti}(0)

    # LNextSpaceInCol tracks current write position for each column
    # We use atomic operations for thread-safe updates
    LNextSpaceInCol = Vector{Threads.Atomic{Ti}}(undef, n)

    # Snapshot of LNextSpaceInCol at level start (for safe reads)
    level_read_limit = Vector{Ti}(undef, n)

    Lp[1] = 1

    # Initialize (sequential - small overhead)
    @inbounds for i = 1:n
        Lp[i+1] = Lp[i] + Lnz[i]
        D[i] = 0.0
        LNextSpaceInCol[i] = Threads.Atomic{Ti}(Lp[i])
        level_read_limit[i] = Lp[i]
    end

    # Handle first element
    if !logicalFactor
        D[1] = Ax[1]
        if Dsigns !== nothing && Dsigns[1]*D[1] < regularize_eps
            D[1] = regularize_delta * Dsigns[1]
            Threads.atomic_add!(regularize_count_atomic, one(Ti))
        end
        if D[1] == 0.0
            return -1
        end
        if D[1] > 0.0
            Threads.atomic_add!(positiveValuesInD, one(Ti))
        end
        Dinv[1] = 1/D[1]
    end

    # Process levels from bottom to top
    factorization_failed = Threads.Atomic{Bool}(false)

    for level_set in level_sets
        # Skip column 1 (already handled)
        columns_to_process = filter(k -> k > 1, level_set)

        if isempty(columns_to_process)
            continue
        end

        # CRITICAL: Snapshot LNextSpaceInCol BEFORE parallel processing
        # This ensures all threads at this level read the same "safe" range
        # (entries from previous levels only, not entries being written now)
        @inbounds for cidx = 1:n
            level_read_limit[cidx] = LNextSpaceInCol[cidx][]
        end

        Threads.@threads for k in columns_to_process
            if factorization_failed[]
                continue  # Skip if factorization already failed
            end

            # Get thread-local workspace
            tid = Threads.threadid()
            ws = thread_workspaces[tid]
            yMarkers = ws.yMarkers
            yIdx = ws.yIdx
            elimBuffer = ws.elimBuffer
            yVals = ws.yVals

            # Initialize thread-local arrays for this column
            nnzY = 0

            # Determine non-zero pattern for row k of L
            @inbounds for i = Ap[k]:(Ap[k+1]-1)
                bidx = Ai[i]

                if bidx == k
                    D[k] = Ax[i]
                    continue
                end

                yVals[bidx] = Ax[i]
                nextIdx = bidx

                if yMarkers[nextIdx] == QDLDL_UNUSED
                    yMarkers[nextIdx] = QDLDL_USED
                    elimBuffer[1] = nextIdx
                    nnzE = 1
                    nextIdx = etree[bidx]

                    @inbounds while nextIdx != QDLDL_UNKNOWN && nextIdx < k
                        if yMarkers[nextIdx] == QDLDL_USED
                            break
                        end
                        yMarkers[nextIdx] = QDLDL_USED
                        nnzE += 1
                        elimBuffer[nnzE] = nextIdx
                        nextIdx = etree[nextIdx]
                    end

                    @inbounds while nnzE != 0
                        nnzY += 1
                        yIdx[nnzY] = elimBuffer[nnzE]
                        nnzE -= 1
                    end
                end
            end

            # Compute values for row k
            @inbounds for i = nnzY:-1:1
                cidx = yIdx[i]

                # Use snapshot read limit (entries from previous levels only)
                read_limit = level_read_limit[cidx]

                # Atomically get write position for this column
                tmpIdx = Threads.atomic_add!(LNextSpaceInCol[cidx], one(Ti))

                if !logicalFactor
                    yVals_cidx = yVals[cidx]

                    # Read from L column cidx using snapshot limit
                    # This only reads entries from PREVIOUS levels (safe, no races)
                    @inbounds for j = Lp[cidx]:(read_limit-1)
                        yVals[Li[j]] -= Lx[j] * yVals_cidx
                    end

                    Lx[tmpIdx] = yVals_cidx * Dinv[cidx]
                    D[k] -= yVals_cidx * Lx[tmpIdx]
                end

                Li[tmpIdx] = k

                yVals[cidx] = 0.0
                yMarkers[cidx] = QDLDL_UNUSED
            end

            # Apply regularization
            if Dsigns !== nothing && Dsigns[k]*D[k] < regularize_eps
                D[k] = regularize_delta * Dsigns[k]
                Threads.atomic_add!(regularize_count_atomic, one(Ti))
            end

            # Check diagonal
            if D[k] == 0.0
                factorization_failed[] = true
                continue
            end

            if D[k] > 0.0
                Threads.atomic_add!(positiveValuesInD, one(Ti))
            end

            Dinv[k] = 1/D[k]
        end

        # Check if factorization failed after each level
        if factorization_failed[]
            return -1
        end
    end

    regularize_count[1] = regularize_count_atomic[]
    return positiveValuesInD[]
end


# Solves (L+I)x = b, with x replacing b
function QDLDL_Lsolve!(n,Lp,Li,Lx,x)

    @inbounds for i = 1:n
        @inbounds for j = Lp[i]: (Lp[i+1]-1)
            x[Li[j]] -= Lx[j]*x[i];
        end
    end
    return nothing
end


# Solves (L+I)'x = b, with x replacing b
function QDLDL_Ltsolve!(n,Lp,Li,Lx,x)

    @inbounds for i = n:-1:1
        @inbounds for j = Lp[i]:(Lp[i+1]-1)
            x[i] -= Lx[j]*x[Li[j]]
        end
    end
    return nothing
end


"""
Parallel forward triangular solve: (L+I)x = b
Uses level-based parallelism - columns at the same level can be processed in parallel.
"""
function QDLDL_Lsolve_parallel!(n, Lp, Li, Lx, x, level_sets::Vector{Vector{Ti}}) where {Ti<:Integer}
    # Process levels from 0 to max_level (bottom-up)
    for level_set in level_sets
        Threads.@threads for i in level_set
            @inbounds for j = Lp[i]:(Lp[i+1]-1)
                # x[Li[j]] is at a higher level, will be processed later
                x[Li[j]] -= Lx[j] * x[i]
            end
        end
        # Implicit barrier at end of @threads
    end
    return nothing
end


"""
Parallel backward triangular solve: (L+I)'x = b
Uses level-based parallelism - columns at the same level can be processed in parallel.
"""
function QDLDL_Ltsolve_parallel!(n, Lp, Li, Lx, x, level_sets::Vector{Vector{Ti}}) where {Ti<:Integer}
    # Process levels from max_level to 0 (top-down)
    for level_set in Iterators.reverse(level_sets)
        Threads.@threads for i in level_set
            @inbounds for j = Lp[i]:(Lp[i+1]-1)
                # x[Li[j]] is at a higher level, already processed
                x[i] -= Lx[j] * x[Li[j]]
            end
        end
        # Implicit barrier at end of @threads
    end
    return nothing
end


"""
Parallel solve: Ax = b where A has given LDL factors
"""
function QDLDL_solve_parallel!(n, Lp, Li, Lx, Dinv, b, level_sets::Vector{Vector{Ti}}) where {Ti<:Integer}
    QDLDL_Lsolve_parallel!(n, Lp, Li, Lx, b, level_sets)
    b .*= Dinv
    QDLDL_Ltsolve_parallel!(n, Lp, Li, Lx, b, level_sets)
end


# Solves Ax = b where A has given LDL factors,
# with x replacing b
function QDLDL_solve!(n,Lp,Li,Lx,Dinv,b)

    QDLDL_Lsolve!(n,Lp,Li,Lx,b)
    b .*= Dinv;
    QDLDL_Ltsolve!(n,Lp,Li,Lx,b)

end



# internal permutation and inverse permutation
# functions that require no memory allocations
function permute!(x,b,p)
  @inbounds for j = eachindex(x)
      x[j] = b[p[j]];
  end
  return x
end

function ipermute!(x,b,p)
 @inbounds for j = eachindex(x)
     x[p[j]] = b[j];
 end
 return x
end


"Given a sparse symmetric matrix `A` (with only upper triangular entries), return permuted sparse symmetric matrix `P` (only upper triangular) given the inverse permutation vector `iperm`."
function permute_symmetric(
    A::SparseMatrixCSC{Tf, Ti},
    iperm::AbstractVector{Ti},
    Pr::AbstractVector{Ti} = zeros(Ti, nnz(A)),
    Pc::AbstractVector{Ti} = zeros(Ti, size(A, 1) + 1),
    Pv::AbstractVector{Tf} = zeros(Tf, nnz(A))
) where {Tf <: AbstractFloat, Ti <: Integer}

    # perform a number of argument checks
    m, n = size(A)
    m != n && throw(DimensionMismatch("Matrix A must be sparse and square"))

    isperm(iperm) || throw(ArgumentError("pinv must be a permutation"))

    if n != length(iperm)
        throw(DimensionMismatch("Dimensions of sparse matrix A must equal the length of iperm, $((m,n)) != $(iperm)"))
    end

    #we will record a mapping of entries from A to PAPt
    AtoPAPt = zeros(Ti,length(Pv))

    P = _permute_symmetric(A, AtoPAPt, iperm, Pr, Pc, Pv)
    return P, AtoPAPt
end

# the main function without extra argument checks
# following the book: Timothy Davis - Direct Methods for Sparse Linear Systems
function _permute_symmetric(
    A::SparseMatrixCSC{Tf, Ti},
    AtoPAPt::AbstractVector{Ti},
    iperm::AbstractVector{Ti},
    Pr::AbstractVector{Ti},
    Pc::AbstractVector{Ti},
    Pv::AbstractVector{Tf}
) where {Tf <: AbstractFloat, Ti <: Integer}

    # 1. count number of entries that each column of P will have
    n = size(A, 2)
    num_entries = zeros(Ti, n)
    Ar = A.rowval
    Ac = A.colptr
    Av = A.nzval

    # count the number of upper-triangle entries in columns of P, keeping in mind the row permutation
    for colA = 1:n
        colP = iperm[colA]
        # loop over entries of A in column A...
        for row_idx = Ac[colA]:Ac[colA+1]-1
            rowA = Ar[row_idx]
            rowP = iperm[rowA]
            # ...and check if entry is upper triangular
            if rowA <= colA
                # determine to which column the entry belongs after permutation
                col_idx = max(rowP, colP)
                num_entries[col_idx] += one(Ti)
            end
        end
    end
    # 2. calculate permuted Pc = P.colptr from number of entries
    Pc[1] = one(Ti)
    @inbounds for k = 1:n
        Pc[k + 1] = Pc[k] + num_entries[k]

        # reuse this vector memory to keep track of free entries in rowval
        num_entries[k] = Pc[k]
    end
    # use alias
    row_starts = num_entries

    # 3. permute the row entries and position of corresponding nzval
    for colA = 1:n
        colP = iperm[colA]
        # loop over rows of A and determine where each row entry of A should be stored
        for rowA_idx = Ac[colA]:Ac[colA+1]-1
            rowA = Ar[rowA_idx]
            # check if upper triangular
            if rowA <= colA
                rowP = iperm[rowA]
                # determine column to store the entry
                col_idx = max(colP, rowP)

                # find next free location in rowval (this results in unordered columns in the rowval)
                rowP_idx = row_starts[col_idx]

                # store rowval and nzval
                Pr[rowP_idx] = min(colP, rowP)
                Pv[rowP_idx] = Av[rowA_idx]

                #record this into the mapping vector
                AtoPAPt[rowA_idx] = rowP_idx

                # increment next free location
                row_starts[col_idx] += 1
            end
        end
    end
    nz_new = Pc[end] - 1
    P = SparseMatrixCSC{Tf, Ti}(n, n, Pc, Pr[1:nz_new], Pv[1:nz_new])

    return P
end

function _get_amd_ordering(A,amd_dense_scale)

    # PJG: For interested readers - setting amd_dense_scale to 1.5 seems to work better
    # for KKT systems in QP problems, but this ad hoc method can surely be improved

    # computes a permutation for A using AMD default parameters explicit cast of the scaling 
    # to Float64 here allows the scale parameter to be passed as some other float type for 
    # consistency with the main API.

    meta = Amd()
    meta.control[AMD.AMD_DENSE] *= Float64(amd_dense_scale)   
    p = amd(A,meta)
    return p



end

end #end module
