.shell rm junk.lis
.output junk.lis


create table run_id(id integer primary key);
delete from run_id;

insert into run_id(id) values( (select max(id) from alde_run) );
-- update run_id set id = 5;


create table perf_stats as
with 
  high_activity as (
    select sr.run_id, round_num, simulation_id,
    CASE
      WHEN assay_score > 1 THEN 1
      -- when assay_score > 0.657676441 then 1
      ELSE 0
    END AS high_activity,
    assay_score,
    r.train_rmse, r.train_r2, r.train_spearman,
    r.validation_rmse, r.validation_r2, r.validation_spearman
    from alde_round_top_variant rv, alde_round r,
      alde_simulation s, alde_sub_run sr
    where rv.round_id = r.id 
      and r.simulation_id = s.id
      and s.sub_run_id = sr.id
      and sr.run_id in (11,10) --(select id from run_id)
      and r.round_num in (5,10)
  ),
  round_sim as (
    select run_id, round_num, simulation_id,
      avg(high_activity) fraction_high_activity,
      avg(assay_score) mean_activity,
      max(assay_score) max_activity,
      avg(train_rmse) train_rmse,
      avg(train_r2) train_r2,
      avg(train_spearman) train_spearm,
      avg(validation_rmse) val_rmse,
      avg(validation_r2) val_r2,
      avg(validation_spearman) val_spearm
    from high_activity ha
    group by ha.run_id, ha.round_num, ha.simulation_id
  ),
  perfs as (
    select run_id, r.name, rs.round_num, avg(rs.fraction_high_activity) mean_fha,
        avg(rs.mean_activity) mean_activity,
        avg(rs.max_activity) max_activity,
        avg(rs.train_rmse) train_rmse,
        avg(rs.train_r2) train_r2,
        avg(rs.train_spearm) train_spearm,
        avg(rs.val_rmse) val_rmse,
        avg(rs.val_r2) val_r2,
        avg(rs.val_spearm) val_spearm
    from round_sim rs, alde_run r
    where rs.run_id = r.id
    group by rs.run_id,r.name,rs.round_num
  ),
  ranks as (
    select
    rank() over (
        partition by round_num
        order by mean_fha desc
        rows between unbounded preceding and unbounded following) as mean_fha_rank,
    rank() over (
        partition by round_num
        order by mean_activity desc
        rows between unbounded preceding and unbounded following) as mean_activity_rank,
    rank() over (
        partition by round_num
        order by max_activity desc
        rows between unbounded preceding and unbounded following) as max_activity_rank,
    rank() over (
        partition by round_num
        order by train_rmse asc
        rows between unbounded preceding and unbounded following) as train_rmse_rank,
    rank() over (
        partition by round_num
        order by train_r2 desc
        rows between unbounded preceding and unbounded following) as train_r2_rank,
    rank() over (
        partition by round_num
        order by train_spearm desc
        rows between unbounded preceding and unbounded following) as train_spearm_rank,
    rank() over (
        partition by round_num
        order by val_rmse asc
        rows between unbounded preceding and unbounded following) as val_rmse_rank,
    rank() over (
        partition by round_num
        order by val_r2 desc
        rows between unbounded preceding and unbounded following) as val_r2_rank,
    rank() over (
        partition by round_num
        order by val_spearm desc
        rows between unbounded preceding and unbounded following) as val_spearm_rank,
    *
  from perfs
  ),
  perf_ranks as (
    select 
      run_id, name, round_num,
      round(mean_fha,3) mean_fha,
      round(mean_fha_rank, 3) as mean_fha_rank,
      round(mean_activity, 3) as mean_activity,
      round(mean_activity_rank, 3) as mean_activity_rank,
      round(max_activity, 3) as max_activity,
      round(max_activity_rank, 3) as max_activity_rank,
      round(train_rmse, 3) as train_rmse,
      round(train_rmse_rank, 3) as train_rmse_rank,
      round(train_r2, 3) as train_r2,
      round(train_r2_rank, 3) as train_r2_rank,
      round(train_spearm, 3) as train_spearm,
      round(train_spearm_rank, 3) as train_spearm_rank,
      round(val_rmse, 3) as val_rmse,
      round(val_rmse_rank, 3) as val_rmse_rank,
      round(val_r2, 3) as val_r2,
      round(val_r2_rank, 3) as val_r2_rank,
      round(val_spearm, 3) as val_spearm,
      round(val_spearm_rank, 3) as val_spearm_rank,
      round((mean_fha_rank*2 + mean_activity_rank*2 + max_activity_rank*2 +
         train_rmse_rank + train_r2_rank + train_spearm_rank +
         val_rmse_rank*1.2 + val_r2_rank*1.2 + val_spearm_rank*1.2),3) as total_rank
    from ranks
  )
select *
from perf_ranks
order by round_num, total_rank
;

.output stdout

