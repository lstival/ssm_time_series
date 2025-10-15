import fev

# Create a task from a dataset stored on Hugging Face Hub
task = fev.Task(
    dataset_path="autogluon/chronos_datasets",
    dataset_config="m4_yearly",
    horizon=24,
    num_windows=2,
)

# # A task consists of multiple rolling evaluation windows
# for window in task.iter_windows():
#     print(window)