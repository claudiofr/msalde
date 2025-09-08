.shell rm junk.lis
-- .output junk.lis

select
    id,
    strategy_name,
    end_round_num,
    mean_fha,
    mean_activity,
    max_activity,
    train_rmse,
    train_r2,
    train_spearm,
    val_rmse,
    val_r2,
    val_spearm
from run_metrics_view;
