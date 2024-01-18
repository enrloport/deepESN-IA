include("ESN.jl")


# df = DataFrame( CSV.File("deepESN-IA_2.0_tanh_fashion_CPU.csv") )
# ttl= "deepESN-IA_2.0_tanh_fashion_CPU"

df = DataFrame( CSV.File("deepESN-IA_2.0_tanh_mnist_GPU.csv") )
ttl= "deepESN-IA_2.0_tanh_mnist_GPU"

# df = DataFrame( CSV.File("deepESN-IA_tanh_mnist_GPU.csv") )
# ttl= "deepESN-IA"

# df = DataFrame( CSV.File("deepESN-IA_Iwin_tanh_mnist_GPU.csv") )
# ttl= "deepESN-IA with Identity matrix as Win"

names(df)

df_LE = df[:,["Layers", "Error", "_runtime"]]

le = Dict(
    i => df_LE[df_LE.Layers .== i, : ]
    for i in 1:10
)

println(ttl)
for i in 10:-1:1
    if i in [1,3,5,8,10]
        println(
            string(i*2000)
            , "\t", string( round((1 - mean(le[i].Error))*100, digits=3)  )
            , "\t", string(round(std(le[i].Error), digits=3))
            , "\ttime: ",string( Time(0) + Second( round(mean(le[i]._runtime)) ) )
            , "\tFPS: ", string( round(70000 / round(mean(le[i]._runtime)), digits=2) )
        )
    end
end


# using StatsPlots

# x = [i for i in 1:10]'
# y = hcat([le[i].Error for i in 1:10 ]...)

# StatsPlots.boxplot(x, y, ylims=(0,0.14), xlabel="Number of layers", ylabel="Error", title=ttl, legend=false, alpha=0.7)



function time_to_fps(st)
    num_img = 70000
    sp1 = split(st, "\n")
    for sp in sp1
        sp2 = split(sp," / ")
        c,g = sp2[1],sp2[2]
        gh,gm,gs = map( x-> parse(Int16, x) , split(g,":"))
        ch,cm,cs = map( x-> parse(Int16, x) , split(c,":"))

        fpsc = round(num_img / (ch*3600 + cm*60 + cs), digits=2)
        fpsg = round(num_img / (gh*3600 + gm*60 + gs), digits=2)
        println(fpsc," / ", fpsg)
    end
end

s = "00:01:30 / 00:00:31
00:03:30 / 00:00:51
00:01:31 / 00:00:27"

time_to_fps(s)



