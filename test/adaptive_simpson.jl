# Loaded by runtests.jl after Test and helper.jl.
@testset "Adaptive composite Simpson with a fixed local n" begin
    # Exact polynomial moments and the first nested-grid evaluation count.
    for n in (2,4,6,8), k in 0:3
        r = adaptive_simpson(x -> x^k,-1.,2.,n;atol=1e-12)
        @test r.value ≈ (2.0^(k+1)-(-1.0)^(k+1))/(k+1) atol=1e-13
        @test r.converged
        @test r.evaluations == 2n+1
        @test r.intervals == [(-1.0,2.0)]
    end
    # A single test agrees with the slide's two composite rules and correction.
    for n in (2,4,6,8)
        coarse = simpson(exp,0.,1.,n)
        fine = simpson(exp,0.,0.5,n)+simpson(exp,0.5,1.,n)
        r = adaptive_simpson(exp,0.,1.,n;atol=1e-15,maxdepth=0)
        @test r.value ≈ fine+(fine-coarse)/15 atol=1e-14
        @test r.estimated_error ≈ abs(fine-coarse)/15 atol=1e-15
        @test !r.converged
        @test r.evaluations == 2n+1
    end
    f(x) = 1/(1+400(x-0.3)^2)
    exact = (atan(14)+atan(6))/20
    for n in (2,4,6,8)
        sampled = Float64[]
        counted(x) = (push!(sampled,x); f(x))
        r = adaptive_simpson(counted,0.,1.,n;atol=1e-8)
        @test r.converged && r.estimated_error <= 1e-8
        @test abs(r.value-exact) < 1e-8
        # Reuse both endpoints and all coarse samples, including across siblings.
        @test r.evaluations == length(sampled) == length(unique(sampled))
        @test r.evaluations == 2n*length(r.intervals)+1
        @test first(r.intervals)[1] == 0.0 && last(r.intervals)[2] == 1.0
        @test all(r.intervals[j][2] == r.intervals[j+1][1]
                  for j in 1:length(r.intervals)-1)
        @test minimum(b-a for (a,b) in r.intervals) < maximum(b-a for (a,b) in r.intervals)
    end
    @test adaptive_simpson(f,0.,1.) == adaptive_simpson(f,0.,1.,2)
    for n in (0,1,3,-2)
        @test_throws ArgumentError adaptive_simpson(exp,0.,1.,n)
    end
    for n in (2,4,8)
        @test !adaptive_simpson(exp,1.,nextfloat(1.),n).converged
    end
    @test_throws DomainError adaptive_simpson(x -> Inf,0.,1.,4)
    # Changing n does not turn the heuristic into a certified error bound.
    bump(x) = abs((x-0.37)/0.01) < 1 ? (1-((x-0.37)/0.01)^2)^2 : 0.0
    missed = adaptive_simpson(bump,0.,1.,2)
    @test missed.converged && missed.value == missed.estimated_error == 0.0
    @test missed.evaluations == 5
end

@testset "Uniform Simpson with cached global refinement" begin
    for k in 0:3
        r = uniform_simpson(x -> x^k,-1.,2.,4;atol=1e-12)
        @test r.value ≈ (2.0^(k+1)-(-1.0)^(k+1))/(k+1) atol=1e-13
        @test r.converged && r.evaluations == 9
    end
    sampled = Float64[]
    f(x) = 1+sin(5x)+1/(1+(1000*(x-0.3))^2)+0.5/(1+(100*(x-0.75))^2)
    exact = 1+(1-cos(5))/5+(atan(700)+atan(300))/1000+(atan(25)+atan(75))/200
    counted(x) = (push!(sampled,x); f(x))
    u = uniform_simpson(counted,0.,1.,4;atol=1e-8)
    a = adaptive_simpson(f,0.,1.,4;atol=1e-8)
    @test u.converged && u.estimated_error <= 1e-8
    @test abs(u.value-exact) < 1e-8 && abs(a.value-exact) < 1e-8
    @test u.evaluations == length(sampled) == length(unique(sampled)) == u.subintervals+1
    @test a.evaluations < u.evaluations
    @test all(row.calls == row.n+1 for row in u.history)
    @test !uniform_simpson(exp,0.,1.,4;atol=1e-15,maxlevels=1).converged
    @test_throws ArgumentError uniform_simpson(exp,0.,1.,3)
    @test_throws ArgumentError uniform_simpson(exp,0.,1.;atol=0)
    @test_throws ArgumentError uniform_simpson(exp,0.,1.;maxlevels=0)
    @test_throws DomainError uniform_simpson(x -> Inf,0.,1.)
end
