# Lecture 3, 18 September 2026
# Instructor figure generation and verification.
# Students use Lecture3_support.jl with ../helper.jl.
# Packages: LinearAlgebra (standard library), Plots.
# Execute this script, or include it and call generate_lecture3_figures().
# Students should use Lecture3_support.jl for the five classroom activities.
# Mathematical node i=0,...,n is stored at Julia index i+1.
include(joinpath(@__DIR__, "..", "helper.jl"))

function generate_lecture3_figures(destination=joinpath(@__DIR__,"..","figures","Lecture3"))
    # %% Activity 1: Global Hermite interpolation of sin(x) on [0,pi/2]
    # Rebuild the three example figures with a Lagrange-error panel at the same nodes.
    hermite_comparison_dir = abspath(destination)
    mkpath(hermite_comparison_dir)
    for m in 2:4
        x = collect(range(0.0,pi/2,length=m))
        c = hermite_coefficients(x,sin.(x),cos.(x))
        dc = [k*c[k+1] for k in 1:length(c)-1]
        rv = maximum(abs(evalpoly(z,c)-sin(z)) for z in x)
        rd = maximum(abs(evalpoly(z,dc)-cos(z)) for z in x)
        grid = range(0.0,pi/2,length=1001)
        agreement = maximum(abs(evalpoly(z,c)-hermite(x,sin.(x),cos.(x),z)) for z in grid)
        @assert max(rv,rd,agreement) < 1e-10
        println((activity=1, nodes=m, coefficients=c, value_residual=rv, slope_residual=rd))
        comparison = plot_hermite_lagrange_sine(m)
        savefig(comparison,joinpath(hermite_comparison_dir,
                                   "Hermite-Lagrange-sine-$m-nodes.pdf"))
    end

    # %% Illustration: Error theorem examples at z=pi/8 (not a node for n=1,2,3)
    # n+1 nodes: compare H_(2n+1) and P_n at exactly the same point.
    # All derivatives of sin(x) have absolute value <=1, so these bounds are valid.
    for n in 1:3
        local nodes = collect(range(0.0,pi/2,length=n+1))
        local z = pi/8
        local omega = prod(z .- nodes)
        local hermite_error = abs(sin(z)-hermite(nodes,sin.(nodes),cos.(nodes),z))
        local lagrange_error = abs(sin(z)-lagrange(nodes,sin.(nodes),z))
        local hermite_bound = omega^2/factorial(2n+2)
        local lagrange_bound = abs(omega)/factorial(n+1)
        @assert hermite_error <= hermite_bound
        @assert lagrange_error <= lagrange_bound
        println((n=n,z=z,hermite_error=hermite_error,hermite_bound=hermite_bound,
                 lagrange_error=lagrange_error,lagrange_bound=lagrange_bound))
    end

    # All generated figures stay inside this repository.
    figure_dir = hermite_comparison_dir
    mkpath(figure_dir)
    default(linewidth=2, guidefontsize=11, tickfontsize=10, legendfontsize=10,
            titlefontsize=13, margin=5Plots.mm)

    # %% Illustration: Central Lagrange basis functions before the Runge example
    # Uses L(nodes,j,z) above, with mathematical indices j=0,...,n.
    let panels = [], grid = range(0.0,1.0,length=10001)
        for (n, colour) in zip((4,10,20),(:blue,:darkorange,:purple))
            nodes = collect(range(0.0,1.0,length=n+1))
            j = n ÷ 2
            values = [L(nodes,j,z) for z in grid]
            nodal_values = [L(nodes,j,z) for z in nodes]
            @assert maximum(abs.(nodal_values .- [i == j ? 1.0 : 0.0 for i in 0:n])) < 1e-12
            @assert maximum(abs.(values .- reverse(values))) < 1e-9
            panel = plot(grid,values;label=false,color=colour,linewidth=2.5,
                title="n = $n, j = $j",xlabel="x",ylabel="basis value",
                xlims=(-0.015,1.015),xticks=0:0.25:1,
                left_margin=10Plots.mm,right_margin=5Plots.mm,
                bottom_margin=10Plots.mm,top_margin=7Plots.mm)
            hline!(panel,[0.0];label=false,color=:gray,linestyle=:dot)
            scatter!(panel,nodes,nodal_values;label=false,markershape=:circle,
                markercolor=:white,markerstrokecolor=:black,markerstrokewidth=1.2,
                markersize=4)
            push!(panels,panel)
            println((basis_degree=n,basis_index=j,max_sampled_abs=maximum(abs.(values))))
        end
        # Independent y-scales show each curve clearly. Compare their tick labels.
        comparison = plot(panels...;layout=(1,3),size=(1500,600),
            titlefontsize=14,guidefontsize=12,tickfontsize=11,legend=false)
        output_dir = figure_dir
        savefig(comparison,joinpath(output_dir,"Lagrange-basis-n4-n10-n20.pdf"))
        savefig(comparison,joinpath(output_dir,"Lagrange-basis-n4-n10-n20.png"))
    end

    # %% Illustration: Central Hermite value and derivative basis functions
    # On the same symmetric nodes, L'_(n,n/2)(1/2)=0, so H=L^2.
    # Evaluate the general definitions and check both values and derivatives.
    let value_panels = [], slope_panels = [], grid = range(0.0,1.0,length=10001)
        for (n, colour) in zip((4,10,20),(:blue,:darkorange,:purple))
            nodes = collect(range(0.0,1.0,length=n+1))
            j = n ÷ 2
            xj = nodes[j+1]
            dLj = sum(1/(xj-nodes[i+1]) for i in 0:n if i != j)
            H = z -> (1-2*(z-xj)*dLj)*L(nodes,j,z)^2
            Hhat = z -> (z-xj)*L(nodes,j,z)^2
            @assert abs(dLj) < 1e-12
            # Complex-step differentiation avoids subtractive cancellation.
            step = 1e-20
            for i in 0:n
                z = nodes[i+1]
                delta = i == j ? 1.0 : 0.0
                @assert abs(H(z)-delta) < 1e-12
                @assert abs(Hhat(z)) < 1e-12
                @assert abs(imag(H(z+im*step))/step) < 1e-9
                @assert abs(imag(Hhat(z+im*step))/step-delta) < 1e-9
            end
            for (basis, panels, kind) in ((H,value_panels,:value),(Hhat,slope_panels,:slope))
                values = basis.(grid)
                reflected = kind == :value ? reverse(values) : -reverse(values)
                @assert maximum(abs.(values.-reflected)) < 1e-10*max(1.0,maximum(abs.(values)))
                panel = plot(grid,values;label=false,color=colour,linewidth=2.5,
                    title="n = $n, j = $j",xlabel="x",ylabel="basis value",
                    xlims=(-0.015,1.015),xticks=0:0.25:1,
                    yformatter=n == 20 ? :scientific : :auto,
                    left_margin=12Plots.mm,right_margin=5Plots.mm,
                    bottom_margin=10Plots.mm,top_margin=7Plots.mm)
                hline!(panel,[0.0];label=false,color=:gray,linestyle=:dot)
                scatter!(panel,nodes,basis.(nodes);label=false,markershape=:circle,
                    markercolor=:white,markerstrokecolor=:black,markerstrokewidth=1.2,
                    markersize=4)
                push!(panels,panel)
                println((hermite_basis=kind,n=n,j=j,max_sampled_abs=maximum(abs.(values))))
            end
        end
        output_dir = figure_dir
        for (panels, name) in ((value_panels,"Hermite-value-basis-n4-n10-n20"),
                               (slope_panels,"Hermite-derivative-basis-n4-n10-n20"))
            comparison = plot(panels...;layout=(1,3),size=(1500,600),
                titlefontsize=14,guidefontsize=12,tickfontsize=11,legend=false)
            savefig(comparison,joinpath(output_dir,name*".pdf"))
            savefig(comparison,joinpath(output_dir,name*".png"))
        end
    end

    # %% Activities 2 and 4: Runge phenomenon versus piecewise linear interpolation
    runge_panels = []
    for n in [10,20]
        x = collect(range(-5.0,5.0,length=n+1))
        y = R.(x)
        grid = range(-5.0,5.0,length=4001)
        pg = [lagrange(x,y,z) for z in grid]
        pl = [piecewise_linear(x,y,z) for z in grid]
        @assert maximum(abs(lagrange(x,y,z)-R(z)) for z in x) < 1e-12
        p = plot(grid,R.(grid); label="R(x)",color=:blue,xlabel="x",ylabel="value",
                 title="n=$n: $(n+1) equally spaced nodes",legend=:bottom)
        plot!(p,grid,pg; label="Global polynomial",color=:darkorange,linestyle=:dash)
        plot!(p,grid,pl; label="Piecewise linear",color=:green)
        scatter!(p,x,y; label="Nodes",color=:black,markersize=3)
        push!(runge_panels,p)
        println((activity=2,n=n,global_error=maximum(abs.(R.(grid).-pg)),
                 piecewise_error=maximum(abs.(R.(grid).-pl))))
    end
    runge_plot = plot(runge_panels...;layout=(1,2),size=(1150,430))
    savefig(runge_plot,joinpath(figure_dir,"Runge-global-piecewise.pdf"))

    # %% Activity 3: Global Hermite interpolation of the Runge function
    # R, dR, hermite, and plot_runge_hermite come from ../helper.jl.
    for n in (10,20)
        local p = plot_runge_hermite(n)
        savefig(p,joinpath(figure_dir,"Runge-Hermite-n$n.pdf"))
    end

    # %% Activity 5: Mesh refinement for sin(x), with exact slopes cos(x)
    ns = [4,8,16,32]
    linear_errors = Float64[]
    hermite_errors = Float64[]
    for n in ns
        x = collect(range(0.0,pi/2,length=n+1))
        y,m = sin.(x),cos.(x)
        grid = range(0.0,pi/2,length=10001)
        eL = maximum(abs(sin(z)-piecewise_linear(x,y,z)) for z in grid)
        eH = maximum(abs(sin(z)-piecewise_hermite(x,y,m,z)) for z in grid)
        h = maximum(diff(x))
        @assert eL <= h^2/8 + 1e-14
        @assert eH <= h^4/384 + 1e-14
        push!(linear_errors,eL)
        push!(hermite_errors,eH)
        println((activity=5,n=n,linear_error=eL,hermite_error=eH))
    end
    println((linear_ratios=linear_errors[1:end-1]./linear_errors[2:end],
             hermite_ratios=hermite_errors[1:end-1]./hermite_errors[2:end]))
    convergence_plot = plot(ns,linear_errors;xscale=:log10,yscale=:log10,
        marker=:circle,label="Piecewise linear",xlabel="Number of intervals n",
        ylabel="Maximum sampled error",color=:blue,xticks=(ns,string.(ns)),
        size=(950,480),legend=:bottomleft)
    plot!(convergence_plot,ns,hermite_errors;marker=:circle,label="Cubic Hermite",color=:darkorange)
    plot!(convergence_plot,ns,(pi/2 ./ ns).^2/8;label="Delta^2 / 8 bound",linestyle=:dash,color=:blue)
    plot!(convergence_plot,ns,(pi/2 ./ ns).^4/384;label="Delta^4 / 384 bound",linestyle=:dash,color=:darkorange)
    savefig(convergence_plot,joinpath(figure_dir,"Piecewise-convergence.pdf"))

    println("All Lecture 3 checks passed. Figures saved to "*figure_dir)
    return figure_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_lecture3_figures()
end
