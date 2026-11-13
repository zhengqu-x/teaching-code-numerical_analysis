@testset "Lecture 9 stationary linear iterations" begin
    A = [4. 1.;2. 3.]
    b = [1.,2.]
    xstar = [0.1,0.6]
    j1 = stationary_solve(A,b;method=:jacobi,maxiter=1,rtol=0,atol=0)
    g1 = stationary_solve(A,b;method=:gauss_seidel,maxiter=1,rtol=0,atol=0)
    @test j1.x ≈ [1/4,2/3]
    @test g1.x ≈ [1/4,1/2]
    @test j1.iterates[:,1] == zeros(2)
    @test stationary_matrix(A;method=:jacobi) ≈ [0. -1/4;-2/3 0.]
    @test stationary_matrix(A;method=:gauss_seidel) ≈ [0. -1/4;0. 1/6]
    for method in (:jacobi,:gauss_seidel)
        r = stationary_solve(A,b;method=method,rtol=1e-12)
        @test r.converged && r.status == :residual
        @test norm(r.x-xstar,Inf) < 1e-11
        @test r.residuals[end] <= r.threshold
        @test size(r.iterates) == (2,r.iterations+1)
        @test length(r.residuals) == length(r.step_norms) == r.iterations+1
        T = stationary_matrix(A;method=method)
        q = opnorm(T,Inf)
        @test q < 1
        for k in 1:min(12,r.iterations)
            error = norm(r.iterates[:,k+1]-xstar,Inf)
            @test error <= q^k*norm(xstar,Inf)+1e-14
            @test error <= q/(1-q)*r.step_norms[k+1]+1e-14
        end
    end
    A4 = [10. -1. 2. 0.;-1. 11. -1. 3.;2. -1. 10. -1.;0. 3. -1. 8.]
    b4 = [6.,25.,-11.,15.]
    for method in (:jacobi,:gauss_seidel)
        r = stationary_solve(A4,b4;method=method,rtol=1e-12)
        @test r.converged
        @test r.x ≈ [1.,2.,-1.,1.] atol=1e-10
        @test r.x ≈ A4\b4 atol=1e-10
    end
    bad = [1. .75 .75;.75 1. .75;.75 .75 1.]
    @test minimum(eigvals(Symmetric(bad))) > 0
    @test maximum(abs.(eigvals(stationary_matrix(bad)))) ≈ 1.5
    divergent = stationary_solve(bad,bad*ones(3);maxiter=20)
    @test !divergent.converged && divergent.iterations == 20
    @test divergent.residuals[end] > divergent.residuals[1]
    @test stationary_solve(bad,bad*ones(3);method=:gauss_seidel).converged
    zero = stationary_solve(A,zeros(2))
    @test zero.converged && zero.iterations == 0 && size(zero.iterates) == (2,1)
    @test !stationary_solve(A,b;maxiter=0).converged
    # A residual stopping flag need not control forward error for an ill-conditioned A.
    scaled = stationary_solve(Diagonal([1.,1e-8]),[1.,1e-8];x0=[1.,0.],rtol=1e-8)
    @test scaled.converged && scaled.iterations == 0
    @test norm(scaled.x-ones(2),Inf) == 1.0
    @test_throws DimensionMismatch stationary_solve(ones(2,3),b)
    @test_throws DimensionMismatch stationary_solve(A,[1.])
    @test_throws DimensionMismatch stationary_solve(A,b;x0=[1.])
    @test_throws ArgumentError stationary_solve([0. 1.;1. 2.],b)
    @test_throws ArgumentError stationary_matrix(A;method=:other)
    @test_throws ArgumentError stationary_solve(A,b;rtol=-1)
    @test_throws ArgumentError stationary_solve(A,b;atol=Inf)
    @test_throws ArgumentError stationary_solve(A,b;maxiter=-1)
    @test_throws ArgumentError stationary_solve(A,[NaN,1.])
    # Graph propagation has a provable fixed point and a strictly dominant system.
    W = [1. 1. 0.;1. 1. 1.;0. 1. 1.]
    P = W ./ sum(W,dims=2)
    alpha = 0.2
    G = Matrix{Float64}(I,3,3)-(1-alpha)*P
    rhs = alpha*[1.,0.,-1.]
    @test opnorm((1-alpha)*P,Inf) ≈ 1-alpha
    for i in 1:3
        @test G[i,i]-sum(abs(G[i,j]) for j in 1:3 if j != i) ≈ alpha
    end
    for method in (:jacobi,:gauss_seidel)
        @test stationary_solve(G,rhs;method=method).x ≈ G\rhs atol=1e-9
    end
end
