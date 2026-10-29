# Run from the repository root: julia --project=. test/runtests.jl
ENV["GKSwstype"] = "100"
using Test
include(joinpath(@__DIR__, "..", "helper.jl"))

# Test plots without opening GUI windows or exporting images.
struct QuietPlots <: AbstractDisplay end
Base.display(::QuietPlots, ::Plots.Plot) = nothing
pushdisplay(QuietPlots())

const CODE_ROOT = normpath(joinpath(@__DIR__, ".."))

function fresh_scope(label)
    scope = Module(gensym(label))
    Core.eval(scope, :(include(path) = Base.include(@__MODULE__, path)))
    return scope
end

@testset "Lecture 2 instructor regression checks" begin
    scope = fresh_scope(:Lecture2Checks)
    Base.include(scope, joinpath(CODE_ROOT, "Lecture2", "lecture2_activities.jl"))
    Base.invokelatest() do
        @test scope.run_lecture2_checks() === nothing
    end
end

@testset "Retained interpolation and timing helpers" begin
    nodes = [0.0,0.5,1.0]
    values = nodes.^2
    @test dd_last_recursive(nodes, values) == dd_last_table(nodes, values) == 1.0
    @test dd_seconds(dd_last_table, nodes, values; batch=1) >= 0
    result = only(dd_cost_experiment([4]))
    @test (result.recursive_updates,result.table_updates) == (15,10)
    @test dd_table_scaling([4,8]) === nothing
    @test_throws ArgumentError dd_cost_experiment([21])
    for count in 2:4
        xs = collect(range(0.0,pi/2,length=count))
        coefficients = hermite_coefficients(xs,sin.(xs),cos.(xs))
        derivative = [k*coefficients[k+1] for k in 1:length(coefficients)-1]
        @test maximum(abs(evalpoly(x,coefficients)-sin(x)) for x in xs) < 1e-11
        @test maximum(abs(evalpoly(x,derivative)-cos(x)) for x in xs) < 1e-11
        for z in range(0.0,pi/2,length=21)
            @test hermite(xs,sin.(xs),cos.(xs),z) ≈ evalpoly(z,coefficients) atol=1e-11
        end
    end
    @test hermite([1.0],[2.0],[3.0],2.0) == 5.0
    @test_throws DimensionMismatch hermite(nodes,values,[1.0],0.3)
    for z in range(0.0,1.0,length=21)
        @test piecewise_hermite(nodes,nodes.^3,3nodes.^2,z) ≈ z^3 atol=1e-14
    end
    for count in 2:4
        @test plot_hermite_sine(count) isa Plots.Plot
        @test plot_hermite_lagrange_sine(count) isa Plots.Plot
    end
end

@testset "Lecture 4 spline regression checks" begin
    scope = fresh_scope(:Lecture4Checks)
    Base.include(scope, joinpath(CODE_ROOT,"Lecture4","lecture4_spline_checks.jl"))
    Base.invokelatest() do
        @test scope.run_lecture4_checks() === nothing
    end
end

@testset "B-splines, barycentric interpolation, and Remez" begin
    for n in (1,2,6), p in 1:4
        knots = uniform_knots(n,p)
        @test length(knots) == n+2p+1
        for x in range(0.0,1.0,length=31)
            basis = [bspline_basis(x,i,p,knots) for i in 1:n+p]
            @test minimum(basis) >= -1e-14
            @test sum(basis) ≈ 1.0 atol=1e-12
        end
    end
    local_knots = collect(0.0:0.25:1.0)
    @test [bspline_basis(x,1,3,local_knots) for x in local_knots] ≈ [0,1/6,2/3,1/6,0]
    @test bspline_basis(-0.1,1,3,local_knots) == 0.0
    @test bspline_basis(1.1,1,3,local_knots) == 0.0
    @test plot_cubic_bspline() isa Plots.Plot
    @test plot_cubic_bspline_basis(2) isa Plots.Plot
    @test plot_cubic_bspline_basis(2;full_support=true) isa Plots.Plot
    @test length(plot_bspline_degrees(6).subplots) == 4

    nodes = chebyshev_roots(8)
    weights = barycentric_weights(nodes)
    values = nodes.^3 .- 2nodes .+ 1
    @test issorted(nodes)
    @test all(abs.(nodes) .< 1)
    for x in vcat(nodes,collect(range(-1.0,1.0,length=41)))
        @test barycentric_value(nodes,values,weights,x) ≈ x^3-2x+1 atol=1e-12
    end

    result = remez_relu7()
    @test result.max_error ≈ 0.0229645310334314 atol=1e-12
    @test result.coefficients[[4,6,8]] == zeros(3)
    @test length(result.extrema) == 9
    @test maximum(abs.(abs.(result.errors).-result.max_error)) < 1e-12
    @test all(result.errors[1:end-1] .* result.errors[2:end] .< 0)
    @test maximum(abs(max(0.0,x)-evalpoly(x,result.coefficients))
                  for x in range(-1.0,1.0,length=10001)) <= result.max_error+1e-12
    @test plot_relu_remez7(result) isa Plots.Plot
end

@testset "Classroom activities run in order and independently" begin
    for (relative_path, count) in (("Lecture2/lecture2_support.jl",6),
                                    ("Lecture3/Lecture3_support.jl",5),
                                    ("Lecture4/Lecture4_support.jl",7),
                                    ("Lecture7/Lecture7_support.jl",9))
        path = joinpath(CODE_ROOT,relative_path)
        source = read(path,String)
        pattern = r"(?ms)^#(?: %% Activity| ACTIVITY) (\d+):[^\n]*\n.*?^[ \t]*#=\n(.*?)^[ \t]*=#"
        blocks = collect(eachmatch(pattern,source))
        @test [parse(Int,m.captures[1]) for m in blocks] == collect(1:count)
        sequential = fresh_scope(:SequentialActivities)
        Base.include(sequential,path)
        @test !isdefined(sequential,:fig)  # Activity-owned binding; Plots exports grid.
        for m in blocks
            @test begin
                Base.include_string(sequential,m.captures[2],path)
                true
            end
            independent = fresh_scope(:IndependentActivity)
            Base.include(independent,path)
            @test begin
                Base.include_string(independent,m.captures[2],path)
                true
            end
        end
    end
end

@testset "Lecture 1 example and compatibility entry point" begin
    first_lecture = fresh_scope(:Lecture1)
    @test Base.include(first_lecture,joinpath(CODE_ROOT,"Lecture1","test.jl")) === nothing
    compatibility = fresh_scope(:HermiteCompatibility)
    Base.include(compatibility,joinpath(CODE_ROOT,"Lecture3","hermite_comparison.jl"))
    @test isdefined(compatibility,:plot_hermite_lagrange_sine)
end

@testset "Lecture 7 quadrature and error control" begin
    # External mathematical references: moments, exact integrals, call counts.
    calls = Ref(0)
    counted_exp(x) = (calls[] += 1; exp(x))
    r = romberg_table(counted_exp,0.,1.;levels=6)
    @test r.evaluations == calls[] == 33
    @test abs(r.table[end,end]-expm1(1.0)) < 1e-13
    @test r.table[2,2] ≈ simpson(exp,0.,1.,2)
    @test isnan(r.table[1,2])
    @test romberg_table(x->x,0.,1.;levels=1).evaluations == 2
    for k in 0:3
        a = adaptive_simpson(x->x^k,-1.,2.;atol=1e-12)
        @test a.value ≈ (2.0^(k+1)-(-1.0)^(k+1))/(k+1) atol=1e-13
        @test a.converged
        @test a.evaluations == 5
    end
    f(x) = 1/(1+400(x-0.3)^2)
    calls[] = 0
    counted_peak(x) = (calls[] += 1; f(x))
    a = adaptive_simpson(counted_peak,0.,1.;atol=1e-8)
    @test a.converged && a.estimated_error <= 1e-8
    @test abs(a.value-(atan(14)+atan(6))/20) < 1e-8
    @test a.evaluations == calls[] == 4length(a.intervals)+1
    @test first(a.intervals)[1] == 0.0 && last(a.intervals)[2] == 1.0
    @test all(a.intervals[i][2] == a.intervals[i+1][1] for i in 1:length(a.intervals)-1)
    @test minimum(b-a for (a,b) in a.intervals) < maximum(b-a for (a,b) in a.intervals)
    @test !adaptive_simpson(exp,0.,1.;atol=1e-15,maxdepth=0).converged
    # A genuine failure case, preserved as a teaching point.
    bump(x) = abs((x-0.37)/0.01) < 1 ? (1-((x-0.37)/0.01)^2)^2 : 0.0
    missed = adaptive_simpson(bump,0.,1.;atol=1e-8)
    @test missed.converged && missed.value == missed.estimated_error == 0.0
    # Adjacent floating-point endpoints cannot be subdivided.
    @test !adaptive_simpson(exp,1.,nextfloat(1.)).converged
    for family in (:legendre,:chebyshev,:laguerre,:hermite), q in 1:5
        rule = gauss_rule(family,q)
        @test all(rule.weights .> 0) && issorted(rule.nodes)
        for k in 0:2q-1
            moment = if family == :legendre
                isodd(k) ? 0.0 : 2/(k+1)
            elseif family == :chebyshev
                isodd(k) ? 0.0 : pi*binomial(k,k÷2)/4.0^(k÷2)
            elseif family == :laguerre
                Float64(factorial(k))
            else
                isodd(k) ? 0.0 : sqrt(pi)*factorial(k)/(4.0^(k÷2)*factorial(k÷2))
            end
            @test sum(rule.weights.*rule.nodes.^k) ≈ moment atol=1e-10 rtol=1e-11
        end
    end
    # Error display blanks unused cells while retaining numerical NaNs internally.
    io = IOBuffer()
    print_romberg_errors([2.0 NaN;1.5 1.6],1.6;io=io)
    printed = String(take!(io))
    @test !occursin("NaN",printed)
    @test occursin("4.0000e-01",printed)
    @test occursin("0.0000e+00",printed)
    hermite_rule = gauss_rule(:hermite,9)
    sigmoid(x) = 1/(1+exp(-x))
    @test sum(hermite_rule.weights.*sigmoid.(sqrt(2).*hermite_rule.nodes))/sqrt(pi) ≈ 0.5 atol=1e-14
    rule = gauss_rule(:legendre,3)
    @test rule.nodes ≈ [-sqrt(3/5),0,sqrt(3/5)] atol=1e-14
    @test rule.weights ≈ [5/9,8/9,5/9] atol=1e-14
    @test abs(sum(rule.weights.*rule.nodes.^6)-2/7) > 0.01
    for k in 0:5
        @test gauss_legendre(x->x^k,2.,5.,3) ≈ (5.0^(k+1)-2.0^(k+1))/(k+1)
    end
    for action in (() -> romberg_table(exp,0.,1.;levels=0),
                   () -> adaptive_simpson(exp,0.,1.;atol=0),
                   () -> adaptive_simpson(x->Inf,0.,1.),
                   () -> gauss_rule(:unknown,3),() -> gauss_rule(:legendre,0))
        @test_throws Exception action()
    end
end

println("All teaching-code tests passed.")
