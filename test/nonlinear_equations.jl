@testset "Lecture 10 nonlinear iterations" begin
    f(x)=x^2-2
    br=bisect_root(f,1.,2.;atol=1e-10)
    @test br.converged && br.status == :interval
    @test abs(BigFloat(br.x)-sqrt(big"2")) <= br.bound
    @test br.bound <= 1e-10
    @test br.calls == br.iterations+3
    @test length(br.iterates)==length(br.bounds)==length(br.residuals)==br.iterations+1
    @test all(diff(br.bounds) .< 0)
    @test br.interval[1] <= sqrt(2) <= br.interval[2]
    @test bisect_root(identity,0.,1.).status == :exact_root
    @test bisect_root(identity,-1.,1.).x == 0
    @test !bisect_root(f,1.,2.;maxiter=0).converged
    @test_throws ArgumentError bisect_root(f,2.,3.)
    @test_throws ArgumentError bisect_root(f,2.,1.)
    @test_throws ArgumentError bisect_root(f,1.,Inf)
    @test_throws ArgumentError bisect_root(x->Inf,1.,2.)
    @test_throws ArgumentError bisect_root(f,1.,2.;atol=0)
    # signbit rather than endpoint products also handles tiny opposite signs.
    tiny=bisect_root(x->1e-300*x,-1.,2.;atol=1e-6)
    @test tiny.converged && abs(tiny.x)<=tiny.bound
    fp=fixedpoint_solve(x->x/2+1,0.;contraction=.5,atol=1e-10)
    @test fp.converged && fp.status == :error_bound && fp.certified
    @test abs(fp.x-2) <= fp.bounds[end]
    @test fp.bounds[end] <= 1e-10
    @test length(fp.iterates)==length(fp.bounds)==length(fp.residuals)==length(fp.step_norms)==fp.iterations+1
    @test all(abs.(fp.iterates.-2) .<= fp.bounds .+ 1e-14)
    @test fp.bounds[end] ≈ .5/(1-.5)*fp.step_norms[end]
    raw=fixedpoint_solve(x->x/2+1,0.)
    @test raw.converged && raw.status == :residual && !raw.certified
    @test all(isinf,raw.bounds)
    exact=fixedpoint_solve(x->2.,0.;contraction=0.)
    @test exact.x==2 && exact.iterations==1 && exact.bounds[end]==0
    @test fixedpoint_solve(identity,2.;contraction=.5).iterations==0
    @test !fixedpoint_solve(x->x+1.,0.;maxiter=5).converged
    @test fixedpoint_solve(x->Inf,0.).status == :nonfinite
    @test_throws ArgumentError fixedpoint_solve(identity,0.;contraction=1.)
    @test_throws ArgumentError fixedpoint_solve(identity,NaN)
    @test_throws DimensionMismatch fixedpoint_solve(x->[1.,2.,3.],[0.,0.])
    W=[.35 .10;-.05 .30]; b=[1.,-.5]; u=.5
    T(z)=tanh.(W*z+u*b)
    F(z)=z-T(z)
    DF(z)=Matrix{Float64}(I,2,2)-Diagonal(1 .- T(z).^2)*W
    vfp=fixedpoint_solve(T,zeros(2);contraction=opnorm(W,Inf),atol=1e-12)
    vn=newton_system(F,DF,zeros(2);atol=1e-12)
    @test vfp.converged && vn.converged
    @test norm(vfp.x-vn.x,Inf) < 2e-12
    @test size(vfp.iterates)==(2,vfp.iterations+1)
    @test size(vn.iterates)==(2,vn.iterations+1)
    @test norm(F(vn.x),Inf)<=1e-12
    nr=newton_solve(f,x->2x,1.;atol=1e-12)
    @test nr.converged && nr.status == :residual
    @test abs(nr.x-sqrt(2))<1e-12
    @test length(nr.iterates)==length(nr.residuals)==length(nr.step_norms)==nr.iterations+1
    @test newton_solve(x->x^2+1,x->2x,0.).status == :zero_derivative
    @test newton_solve(x->Inf,x->1.,0.).status == :nonfinite
    @test newton_solve(x->x+1,x->Inf,0.).status == :nonfinite
    cycle=newton_solve(x->x^3-2x+2,x->3x^2-2,0.;maxiter=6)
    @test !cycle.converged && cycle.status == :maxiter
    @test cycle.iterates == [0.,1.,0.,1.,0.,1.,0.]
    @test newton_solve(identity,x->1.,0.).iterations==0
    @test !newton_solve(f,x->2x,1.;maxiter=0).converged
    @test_throws ArgumentError newton_solve(f,x->2x,Inf)
    @test_throws ArgumentError newton_solve(f,x->2x,1.;atol=-1)
    singular=newton_system(x->[x[1]+1,x[2]+1],x->zeros(2,2),zeros(2))
    @test !singular.converged && singular.status == :singular_jacobian
    @test_throws DimensionMismatch newton_system(x->[1.],x->ones(2,2),zeros(2))
    @test_throws DimensionMismatch newton_system(x->x.+1,x->ones(1,1),zeros(2))
    @test_throws ArgumentError newton_system(identity,identity,Float64[])
    F3(x)=[3x[1]-cos(x[2]*x[3])-.5,
        x[1]^2-81(x[2]+.1)^2+sin(x[3])+1.06,
        exp(-x[1]*x[2])+20x[3]+(10pi-3)/3]
    J3(x)=[3. x[3]*sin(x[2]*x[3]) x[2]*sin(x[2]*x[3]);
        2x[1] -162(x[2]+.1) cos(x[3]);
        -x[2]*exp(-x[1]*x[2]) -x[1]*exp(-x[1]*x[2]) 20.]
    root=[.5,0.,-pi/6]
    n3=newton_system(F3,J3,[.1,.1,-.1])
    @test n3.converged && norm(n3.x-root,Inf)<1e-10
    @test norm(F3(n3.x),Inf)<1e-12
    # Compare the analytic Jacobian with a meaningful central-difference check.
    x=[.2,.1,-.3]; h=1e-6
    numerical=hcat([(F3(x+h*[j==i ? 1. : 0. for j in 1:3])-F3(x-h*[j==i ? 1. : 0. for j in 1:3]))/(2h) for i in 1:3]...)
    @test norm(J3(x)-numerical,Inf)<1e-7
end
