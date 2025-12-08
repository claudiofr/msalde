select id, config_id, name, dataset_name, start_ts, end_ts
from alde_run
where end_ts is not null
order by end_ts;
