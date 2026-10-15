# Lecture 1: Newton iteration for sqrt(2), in Float64 and BigFloat.
# Run this file or select one precision example at a time.
using Printf
println("Float64:")
let x = 1.0, xstar = sqrt(2.0)
for k = 0:10
    e = abs(x - xstar)
    @printf("k = %2d,  error = %.5e\n", k, e)
    x = (x + 2/x) / 2
end
end

#=
println()
println("BigFloat:")

setprecision(BigFloat, 256) do
    x = BigFloat(1);
    xstar = sqrt(BigFloat(2));
    for k = 0:10
        e = abs(x - xstar)
        @printf("k = %2d,  error = %.5e\n", k, e)
        x = (x + BigFloat(2)/x) / BigFloat(2)
    end
end
=#
