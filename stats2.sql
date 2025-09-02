.shell rm junk.lis
.output junk.lis

create table perf_stats(
  run_id integer primary key,
  round_num integer primary key,
  frac_high_activity real,
  mean_activity real,
  max_activity real,
  train_rmse real,
  train_r2 real,
  train_spearm real,
  val_rmse real,
  val_r2 real,
  val_spearm real,
);

create table run_id(id integer primary key);
delete from run_id;

insert into run_id(id) values( (select max(id) from run_id) );
-- update run_id set id = 5;


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
      and sr.run_id = (select id from run_id)
      and r.round_num in (5,10)
  ),
  round_sim as (
    select run_id, round_num, simulation_id,
      avg(high_activity) fraction_high_activity,
      avg(assay_score) mean_activity,
      max(assay_score) max_activity,
      avg(r.train_rmse) train_rmse,
      avg(r.train_r2) train_r2,
      avg(r.train_spearman) train_spearm,
      avg(r.validation_rmse) val_rmse,
      avg(r.validation_r2) val_r2,
      avg(r.validation_spearman) val_spearm
    from high_activity ha
    group by ha.run_id, ha.round_num, ha.simulation_id
  ),
  perfs as (
    select run_id, r.name, rs.round_num, avg(rs.fraction_high_activity) mean_fha,
        avg(rs.mean_activity) mean_activity,
        avg(rs.max_activity) max_activity,
        avg(r.train_rmse) train_rmse,
        avg(r.train_r2) train_r2,
        avg(r.train_spearman) train_spearm,
        avg(r.validation_rmse) val_rmse,
        avg(r.validation_r2) val_r2,
        avg(r.validation_spearman) val_spearm
    from round_sim rs, alde_run r
    where rs.run_id = r.id
    group by rs.round_num
  ),
  with ranks as (
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
  ), final as (
    select 
      run_id, name, round_num,
      mean_fha, mean_fha_rank,
      mean_activity, mean_activity_rank,
      max_activity, max_activity_rank,
      train_rmse, train_rmse_rank,
      train_r2, train_r2_rank,
      train_spearm, train_spearm_rank,
      val_rmse, val_rmse_rank,
      val_r2, val_r2_rank,
      val_spearm, val_spearm_rank,
      (mean_fha_rank + mean_activity_rank + max_activity_rank +
         train_rmse_rank + train_r2_rank + train_spearm_rank +
         val_rmse_rank + val_r2_rank + val_spearm_rank) as total_rank
    from with ranks
  )
select *
from final
order by round_num, total_rank
;


order by run_id, rs.round_num;

.output stdout

