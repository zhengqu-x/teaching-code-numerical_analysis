# Lecture 4: instructor checks for the postponed cubic-spline material.
# Students use Lecture4_support.jl with ../helper.jl; this verifies Activities 1-3.
include(joinpath(@__DIR__, "..", "helper.jl"))

# Instructor-only derivative checks; students do not need these in helper.jl.
function local_coefficients(x,y,m,i)
    h = x[i+1]-x[i]
    d = (y[i+1]-y[i])/h
    return [y[i],m[i],(3d-2m[i]-m[i+1])/h,
            (m[i]+m[i+1]-2d)/h^2]
end

function local_value(x,y,m,i,z,r=0)
    c = local_coefficients(x,y,m,i)
    for _ in 1:r
        c = [k*c[k+1] for k in 1:length(c)-1]
    end
    return evalpoly(z-x[i],c)
end

function spline_checks(x,y,m)
    n = length(x)-1
    value_residual = maximum(max(abs(local_value(x,y,m,i,x[i])-y[i]),
                                abs(local_value(x,y,m,i,x[i+1])-y[i+1]))
                             for i in 1:n)
    jumps = [maximum([0.0; [abs(local_value(x,y,m,i,x[i+1],r)-
                               local_value(x,y,m,i+1,x[i+1],r)) for i in 1:n-1]])
             for r in 1:2]
    boundary_second = (local_value(x,y,m,1,x[1],2),
                       local_value(x,y,m,n,x[end],2))
    return (value_residual=value_residual, first_jump=jumps[1],
            second_jump=jumps[2], boundary_second=boundary_second)
end

function run_lecture4_checks(; output_dir=nothing)
    # Passing output_dir enables figure export; checks alone do not write files.
    output_dir === nothing || mkpath(output_dir)
    default(linewidth=2, guidefontsize=11, tickfontsize=10, legendfontsize=10,
            titlefontsize=13, margin=5Plots.mm)

    # %% Activity 1: Clamped spline through (0,0), (1,1), (2,0)
    x, y = [0.0,1.0,2.0], [0.0,1.0,0.0]
    m = spline_slopes(x,y;boundary=:clamped,endpoint_slopes=(1.0,1.0))
    @assert isapprox(m,[1.0,-0.5,1.0];atol=1e-14)
    checks = spline_checks(x,y,m)
    @assert max(checks.value_residual,checks.first_jump,checks.second_jump) < 1e-12
    println((activity=1,slopes=m,checks=checks))

    # %% Activity 2: Natural cubic spline through (0,0), (1,1), (2,0)
    x = [0.0,1.0,2.0]
    y = [0.0,1.0,0.0]
    m = spline_slopes(x,y)
    @assert isapprox(m,[1.5,0.0,-1.5];atol=1e-14)
    checks = spline_checks(x,y,m)
    @assert max(checks.value_residual,checks.first_jump,checks.second_jump,
                maximum(abs.(checks.boundary_second))) < 1e-12
    println((activity=2,slopes=m,checks=checks))
    # Compare the implementation with the explicit hand-calculated answer.
    exact_spline(z) = z <= 1 ? 1.5z-0.5z^3 : 1.5*(2-z)-0.5*(2-z)^3
    @assert maximum(abs(piecewise_hermite(x,y,m,z)-exact_spline(z))
                    for z in range(0.0,2.0,length=501)) < 1e-12

    # %% Activity 3: Natural versus clamped boundaries on [0,pi/2]
    x = collect(range(0.0,pi/2,length=5))
    y = sin.(x)
    mn = spline_slopes(x,y)
    mc = spline_slopes(x,y;boundary=:clamped,endpoint_slopes=(1.0,0.0))
    grid = range(0.0,pi/2,length=2001)
    sn = [piecewise_hermite(x,y,mn,z) for z in grid]
    sc = [piecewise_hermite(x,y,mc,z) for z in grid]
    p1 = plot(grid,sin.(grid);label="sin(x)",color=:blue,xlabel="x",ylabel="value",
              title="Five equally spaced nodes",legend=:bottomright)
    plot!(p1,grid,sn;label="Natural spline",color=:darkorange,linestyle=:dash)
    plot!(p1,grid,sc;label="Clamped spline",color=:green,linestyle=:dot)
    scatter!(p1,x,y;label="Nodes",color=:black,markersize=4)
    p2 = plot(grid,sin.(grid).-sn;label="sin(x) - natural",color=:darkorange,
              xlabel="x",ylabel="signed error",title="Boundary conditions affect the error",
              legend=:topleft)
    plot!(p2,grid,sin.(grid).-sc;label="sin(x) - clamped",color=:green)
    hline!(p2,[0.0];label=false,color=:gray,linestyle=:dot)
    for p in (p1,p2)
        plot!(p;xticks=([0,pi/4,pi/2],["0","pi/4","pi/2"]))
    end
    boundary_plot = plot(p1,p2;layout=(1,2),size=(1150,430))
    if output_dir !== nothing
        savefig(boundary_plot,joinpath(output_dir,"Sine-spline-boundaries.pdf"))
    end
    println((activity=3,natural_error=maximum(abs.(sin.(grid).-sn)),
             clamped_error=maximum(abs.(sin.(grid).-sc))))

    # %% Additional checks: nonuniform nodes, endpoint cases, and polynomial reproduction
    let
    for x in ([0.0,1.0], [0.0,0.07,0.35,0.8,1.0])
        for boundary in (:natural,:clamped)
            # Every affine function satisfies natural conditions.
            y = 2 .+ 3 .* x
            m = spline_slopes(x,y;boundary=boundary,endpoint_slopes=(3.0,3.0))
            @assert maximum(abs.(m.-3)) < 1e-12
            # A clamped cubic is reproduced on a nonuniform mesh.
            if boundary == :clamped
                y = x.^3 .- 2 .* x.^2 .+ x .+ 1
                m = spline_slopes(x,y;boundary=:clamped,endpoint_slopes=(1.0,0.0))
                c = spline_checks(x,y,m)
                @assert max(c.value_residual,c.first_jump,c.second_jump) < 1e-10
                @assert maximum(abs(piecewise_hermite(x,y,m,z)-(z^3-2z^2+z+1))
                                for z in range(0.0,1.0,length=501)) < 1e-12
            end
        end
    end
    end
    # Counterexample to a boundary-free O(Delta^4)*norm(f'''') claim:
    x = collect(range(0.0,1.0,length=5))
    y = x.^2
    mn = spline_slopes(x,y)
    mc = spline_slopes(x,y;boundary=:clamped,endpoint_slopes=(0.0,2.0))
    grid = range(0.0,1.0,length=1001)
    en = maximum(abs(z^2-piecewise_hermite(x,y,mn,z)) for z in grid)
    ec = maximum(abs(z^2-piecewise_hermite(x,y,mc,z)) for z in grid)
    @assert en > 1e-4
    @assert ec < 1e-12
    println((quadratic_natural_error=en,quadratic_clamped_error=ec))
    println("All Lecture 4 spline checks passed.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_lecture4_checks(output_dir=joinpath(@__DIR__,"..","figures","Lecture4"))
end
