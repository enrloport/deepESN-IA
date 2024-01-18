import pandas as pd
import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>

# project_name = "deepESN-IA_tanh_mnist_GPU"
# project_name = "deepESN-IA_Iwin_tanh_mnist_GPU"

project_name = "deepESN-IA_2.0_tanh_fashion_CPU"
# project_name = "deepESN-IA_2.0_tanh_mnist_GPU"

runs = api.runs("elortiz/" + project_name)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
            if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

sum_df = pd.DataFrame.from_records(summary_list)

sum_df.to_csv(project_name + ".csv")



