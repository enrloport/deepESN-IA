include("ESN.jl")


df = DataFrame( CSV.File("deepESN-IA_tanh_mnist_GPU.csv") )
ttl= "deepESN-IA"

df = DataFrame( CSV.File("deepESN-IA_Iwin_tanh_mnist_GPU.csv") )
ttl= "deepESN-IA with Identity matrix as Win"

names(df)

df_LE = df[:,[:Layers, :Error]]

le = Dict(
    i => df_LE[df_LE.Layers .== i, : ]
    for i in 1:10
)


using StatsPlots

x = [i for i in 1:10]'
y = hcat([le[i].Error for i in 1:10 ]...)

StatsPlots.boxplot(x, y, xlabel="Number of layers", ylabel="Error", title=ttl, legend=false, alpha=0.7)

