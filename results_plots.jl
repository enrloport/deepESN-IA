include("ESN.jl")


df = DataFrame( CSV.File("deepESN-IA_tanh_mnist_GPU.csv") )
ttl= "deepESN-IA"

# df = DataFrame( CSV.File("deepESN-IA_Iwin_tanh_mnist_GPU.csv") )
# ttl= "deepESN-IA with Identity matrix as Win"

names(df)
println([n for n in names(df)])

df_LE = df[:,["Layers", "Error", "_runtime"]]

le = Dict(
    i => df_LE[df_LE.Layers .== i, : ]
    for i in 1:10
)

for i in 1:10
    if i in [1,3,5,8,10]
        println(string(i*2000), "\t", string((1 - mean(le[i].Error))*100), "\t", string(std(le[i].Error)), "\ttime: ", string(mean(le[i]._runtime))  )
    end
end

using StatsPlots

x = [i for i in 1:10]'
y = hcat([le[i].Error for i in 1:10 ]...)

StatsPlots.boxplot(x, y, ylims=(0,0.14), xlabel="Number of layers", ylabel="Error", title=ttl, legend=false, alpha=0.7)

