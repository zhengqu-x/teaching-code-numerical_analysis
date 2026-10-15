# Lecture 3: classroom support (18 September 2026).
# In Julia, from this folder: include("Lecture3_support.jl")
# Load ../helper.jl and select the optional activities below.
# Cubic spline systems and their activities were postponed to Lecture 4.
# They are now in ../Lecture4/Lecture4_support.jl (Activities 1-3).
# Packages: LinearAlgebra (standard library), Plots.
# Install dependencies once from the repository root as described in README.md.
# Nodes must be distinct and increasing. Mathematical j=0,...,n is stored at j+1.
# Shared functions and package imports; the path is relative to this file.
include(joinpath(@__DIR__, "..", "helper.jl"))

# CLASSROOM ACTIVITIES
# For commented blocks, select the code inside #= ... =# and run it,
# or copy it into the REPL, after loading this file. Blocks are independent.
# Slide order: 1 sin(x); 2 Lagrange Runge; 3 Hermite Runge;
# 4 piecewise linear Runge; 5 piecewise Hermite refinement.

# %% Activity 1: Hermite and Lagrange interpolation of sin(x)
# Try n=1, 2, 3: H_(2n+1) and P_n use the same n+1 nodes.
#=
n = 1
xx = collect(range(0.0,pi/2,length=n+1))
grid = range(0.0,pi/2,length=1001)
hv = [hermite(xx,sin.(xx),cos.(xx),z) for z in grid]
pv = [lagrange(xx,sin.(xx),z) for z in grid]
p = plot(grid,sin.(grid);label="sin(x)",xlabel="x",ylabel="value")
plot!(p,grid,hv;label="Hermite",linestyle=:dash)
plot!(p,grid,pv;label="Lagrange",linestyle=:dot)
scatter!(p,xx,sin.(xx);label="Nodes",color=:black)
display(p)
(maximum(abs.(sin.(grid).-hv)), maximum(abs.(sin.(grid).-pv)))
=#

# %% Activity 2: Lagrange interpolation of the Runge function
# After the Runge plots and AI discussion. Try n=10 and n=20.
#=
n = 10
xx = collect(range(-5.0,5.0,length=n+1))
yy = R.(xx)
grid = range(-5.0,5.0,length=4001)
pv = [lagrange(xx,yy,z) for z in grid]
p = plot(grid,R.(grid);label="R(x)",xlabel="x",ylabel="value")
plot!(p,grid,pv;label="Lagrange",linestyle=:dash)
scatter!(p,xx,yy;label="Nodes",color=:black)
display(p)
nodal_residual = maximum(abs(lagrange(xx,yy,z)-R(z)) for z in xx)
sampled_error = maximum(abs.(R.(grid).-pv))
(nodal_residual,sampled_error)
=#

# %% Activity 3: Hermite interpolation of the Runge function
# Repeat with n=20. Does matching slopes remove the endpoint oscillations?
#=
 n = 10
 xx = collect(range(-5.0,5.0,length=n+1))
 yy, dy = R.(xx), dR.(xx)
 grid = range(-5.0,5.0,length=4001)
 hv = [hermite(xx,yy,dy,z) for z in grid]
 p = plot(grid,R.(grid); label="R(x)",xlabel="x",ylabel="value")
 plot!(p,grid,hv; label="Hermite",linestyle=:dash)
 scatter!(p,xx,yy; label="Nodes",color=:black)
 display(p)
 maximum(abs.(R.(grid).-hv))  # sampled error, not the exact supremum
 display(plot_runge_hermite(20))
 =#

# %% Activity 4: Piecewise linear interpolation of the Runge function
# After introducing piecewise interpolation. Try n=10, 20, 30 (11,21,31 nodes).
#=
n = 10
xx = collect(range(-5.0,5.0,length=n+1))
yy = R.(xx)
grid = range(-5.0,5.0,length=4001)
pl = [piecewise_linear(xx,yy,z) for z in grid]
p = plot(grid,R.(grid);label="R(x)",xlabel="x",ylabel="value")
plot!(p,grid,pl;label="Piecewise linear",linestyle=:dash)
scatter!(p,xx,yy;label="Nodes",color=:black)
display(p)
maximum(abs.(R.(grid).-pl))
=#

# %% Activity 5: Piecewise linear versus piecewise cubic Hermite
# Compare n=4, 8, 16, 32. Exact slopes are available for this example.
#=
let previous_linear = NaN, previous_hermite = NaN
for n in (4,8,16,32)
    xx = collect(range(0.0,pi/2,length=n+1))
    grid = range(0.0,pi/2,length=4001)
    pl = [piecewise_linear(xx,sin.(xx),z) for z in grid]
    ph = [piecewise_hermite(xx,sin.(xx),cos.(xx),z) for z in grid]
    eL = maximum(abs.(sin.(grid).-pl))
    eH = maximum(abs.(sin.(grid).-ph))
    println((n=n,linear_error=eL,hermite_error=eH,
             linear_ratio=previous_linear/eL,hermite_ratio=previous_hermite/eH))
    previous_linear, previous_hermite = eL,eH
end
end
=#
