# Function to test an already trained deepESNIA struct
function __do_test_deepESNIA2_mnist!(deepE, args::Dict)
    test_length = args[:test_length]

    classes_Y    = Array{Tuple{Float64,Int,Int}}[]
    wrong_class  = []
    deepE.Y        = []

    # if args[:gpu]
    #     # deepE.Y = CuArray(deepE.Y)
    #     f = (u) -> CuArray(reshape(u, :, 1))
    # else
    #     f = (u) -> reshape(u, :, 1)
    # end

    f = args[:gpu] ? (u) -> CuArray(reshape(u, :, 1)) : (u) -> reshape(u, :, 1)

    for t in 1:test_length
                
        fu = f(args[:test_data][:,:,t])
        __update(deepE.esns[1], args[:test_data][:,:,t], f )

        for i in 2:length(deepE.esns)
            __update(deepE.esns[i], vcat(deepE.esns[i-1].x, fu ), f )
        end

        x = vcat(f(args[:test_data][:,:,t]), [_e.x for _e in deepE.esns]..., f([1]) )
        # x = vcat(f(args[:test_data][:,:,t]), deepE.esns[end].x, f([1]) )


        pairs  = []
        for c in args[:classes]
            yc = Array(deepE.classes_Routs[c] * x)[1]
            push!(pairs, (yc, c, args[:test_labels][t]))
        end
        pairs_sorted  = reverse(sort(pairs))
        
        if pairs_sorted[1][2] != pairs_sorted[1][3]
            push!(wrong_class, (args[:test_data][t], pairs_sorted[1], pairs_sorted[2], t ) ) 
        end

        push!(deepE.Y,[pairs_sorted[1][2] ;])
        push!(classes_Y, pairs )

        for es in deepE.esns
            es.x[:] = es.x .* 0
        end
    end

    deepE.wrong_class= wrong_class
    deepE.classes_Y  = classes_Y
    deepE.Y_target   = args[:test_labels]
    deepE.error      = length(wrong_class) / length(classes_Y)

    # println(length(wrong_class))
    # println(length(classes_Y))

end
