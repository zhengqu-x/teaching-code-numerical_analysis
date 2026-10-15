# Numerical Analysis, Lecture 2 -- 11 September 2026
# Instructor reference for the "Predict -> Ask Your AI -> Verify" activities.
# Students should first make a prediction and inspect their AI-generated code.
#
# From the repository folder, load the reference functions with:
#   include("Lecture2/lecture2_activities.jl")
# Run all checks with:
#   run_lecture2_checks()
#
# Shared functions are loaded from ../helper.jl (LinearAlgebra and Plots).
# Mathematical node x_i corresponds to Julia entry xx[i+1].

include(joinpath(@__DIR__, "..", "helper.jl"))
using Test

function run_lecture2_checks()
    @testset "Lecture 2: formulas, indexing, and edge cases" begin
        xx = [-1.0, 0.0, 1.0]
        yy = xx.^2
        V = [x^j for x in xx, j in 0:2]
        @test V \ yy == [0.0, 0.0, 1.0]
        @test det(V) == 2.0
        repeated = [0.0, 0.0, 1.0]
        @test det([x^j for x in repeated, j in 0:2]) == 0.0
        B = [lagrange_basis(xx,j,z) for z in xx, j in 1:3]
        @test B == Matrix{Float64}(I,3,3)
        @test sum(lagrange_basis(xx,j,0.5) for j in 1:3) == 1.0
        @test lagrange(xx,yy,0.5) == 0.25
        cc = divdiff(xx,yy)
        @test cc == [1.0,-1.0,1.0]
        @test newton_value(xx,cc,0.5) == 0.25
        @test divdiff([-1.0,0.0,1.0,2.0],[1.0,0.0,1.0,4.0]) ==
              [1.0,-1.0,1.0,0.0]
        @test divdiff([0.0,1.0,2.0],[1.0,2.0,5.0]) == [1.0,1.0,1.0]
        for z in range(-1,1,length=21)
            @test isapprox(lagrange(xx,yy,z), z^2; atol=1e-14)
            @test isapprox(newton_value(xx,cc,z), z^2; atol=1e-14)
        end
        # A non-quadratic example checks all divided-difference orders.
        nodes = [-2.0,-0.5,0.25,1.0,2.0]
        f(z) = 2z^4 - 3z^3 + z - 7
        dd = divdiff(nodes,f.(nodes))
        for z in range(-2,2,length=31)
            @test isapprox(lagrange(nodes,f.(nodes),z), f(z); atol=1e-12)
            @test isapprox(newton_value(nodes,dd,z), f(z); atol=1e-12)
        end
        # Divided differences are symmetric under permutations of the nodes.
        permutation = [3,5,1,4,2]
        @test isapprox(divdiff(nodes[permutation],f.(nodes[permutation]))[end],
                       dd[end]; atol=1e-12)
        @test lagrange([2.0],[3.0],0.0) == 3.0
        @test divdiff([2.0],[3.0]) == [3.0]
        @test newton_value([2.0],[3.0],0.0) == 3.0
        @test_throws ArgumentError lagrange(repeated,yy,0.5)
        @test_throws ArgumentError divdiff(repeated,yy)
        @test_throws DimensionMismatch lagrange(xx,[1.0],0.5)
        @test_throws BoundsError lagrange_basis(xx,0,0.5)
        @test_throws ArgumentError divdiff(Float64[],Float64[])

        xs = [0.0,pi/4,pi/2]
        z = pi/6
        err = abs(sin(z)-lagrange(xs,sin.(xs),z))
        bound = abs(prod(z .- xs))/factorial(3)
        @test isapprox(bound, pi^3/1296)
        @test err <= bound

        H(z) = 3z^2-2z^3
        dH(z) = 6z-6z^2
        @test [H(0),H(1),dH(0),dH(1)] == [0,1,0,0]
        @test H(0.5) == 0.5
        HL(z) = z^2*(z-1)
        HR(z) = (z-1)*(2-z)^2
        dHL(z) = 3z^2-2z
        dHR(z) = 3z^2-10z+8
        ddHL(z) = 6z-2
        ddHR(z) = 6z-10
        @test [HL(0),HL(1),HR(1),HR(2)] == [0,0,0,0]
        @test [dHL(0),dHL(1),dHR(1),dHR(2)] == [0,1,1,0]
        @test (ddHL(1),ddHR(1)) == (4,-4)
        for z in xx
            @test piecewise_linear(xx,yy,z) == z^2
        end
        @test piecewise_linear(xx,yy,0.5) == 0.5
        @test_throws DomainError piecewise_linear(xx,yy,2.0)
        @test_throws ArgumentError piecewise_linear(reverse(xx),yy,0.0)
    end

    xs = [0.0,pi/4,pi/2]
    z = pi/6
    println("Sine example (actual error, bound): ",
            (abs(sin(z)-lagrange(xs,sin.(xs),z)), abs(prod(z .- xs))/6))
    println("\nGlobal polynomial interpolation of the Runge function:")
    for n in (10,20)
        println(runge_errors(n))
    end
    println("\nOptional: piecewise-linear interpolation:")
    for n in (10,20,40)
        println(runge_errors(n; piecewise=true))
    end
    println("\nOptional: 256-bit BigFloat check of degree 20:")
    setprecision(BigFloat,256) do
        println(runge_errors(20; high_precision=true))
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_lecture2_checks()
end
