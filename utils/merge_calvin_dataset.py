from lerobot.datasets.aggregate import aggregate_datasets

dataset1 = "/remote-home/wukehao/datasets/calvin_task_ABC_D/splitA"
dataset2 = "/remote-home/wukehao/datasets/calvin_task_ABC_D/splitB"
dataset3 = "/remote-home/wukehao/datasets/calvin_task_ABC_D/splitC"
aggregate_datasets(
    repo_ids=['splitA', 'splitB', 'splitC'],
    aggr_repo_id='calvin_merge_ABC',
    roots =[dataset1, dataset2, dataset3],
    aggr_root = "/remote-home/wukehao/datasets/calvin_merge_ABC",
    data_files_size_in_mb = 200,
    video_files_size_in_mb = 500
)