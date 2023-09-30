function __update(esn::ESN, u::Mtx, f::Function)
    # println("R_in type: ", typeof(esn.R_in), " - R_in size: ", size(esn.R_in) )
    # println("f(u) type: ", typeof(f(u)),     " - f(u) size: ", size(f(u))     )
    # println("R type: "   , typeof(esn.R),    " - R size: "   , size(esn.R)    )
    # println("x type: "   , typeof(esn.x),    " - x size: "   , size(esn.x)    )
    # println("size R_in * f(u): ", size(esn.F_in(f,u)), " - ", size(esn.R_in * f(u)) )
    # println("size R * x: ", size(esn.R*esn.x), "\n")

    esn.x[:] = (1-esn.alpha).*esn.x .+ esn.alpha.*esn.sgmd.( esn.F_in(f,u) .+ esn.R*esn.x)
end