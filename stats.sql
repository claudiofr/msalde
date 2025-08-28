select 'mean activity of top n mutants per round with std';

with 
  round_sim as (
    select round_num, simulation_id, avg(assay_score) mean_score
    from alde_round_top_variant rv, alde_round r,
      alde_simulation s, alde_sub_run sr
    where rv.round_id = r.id
      and r.simulation_id = s.id
      and s.sub_run_id = sr.id
      and sr.run_id = (select max(id) from alde_run)
    group by r.id, r.simulation_id
  ),
  round_avg as (
    select round_num, avg(mean_score) mean_score
    from round_sim
    group by round_num
  )
select rs.round_num, avg(rs.mean_score) mean_score,
  SQRT(AVG((rs.mean_score - ra.mean_score) * 
  (rs.mean_score - ra.mean_score))) AS stddev
from round_sim rs, round_avg ra
where rs.round_num = ra.round_num
group by rs.round_num
order by rs.round_num;



select 'fraction high activity mutants by round with std';

with 
  high_activity as (
    select round_num, simulation_id,
    CASE
      -- WHEN assay_score > 1 THEN 1
      when assay_score > 0.657676441 then 1
      ELSE 0
    END AS high_activity
    from alde_round_top_variant rv, alde_round r,
      alde_simulation s, alde_sub_run sr
    where rv.round_id = r.id 
      and r.simulation_id = s.id
      and s.sub_run_id = sr.id
      and sr.run_id = (select max(id) from alde_run)
  ),
  round_sim as (
    select round_num, simulation_id,
      avg(high_activity) fraction_high_activity
    from high_activity ha
    group by ha.round_num, ha.simulation_id
  ),
  round_avg as (
    select round_num, avg(fraction_high_activity) mean_fha
    from round_sim
    group by round_num
  )
select rs.round_num, avg(rs.fraction_high_activity) mean_fha,
  SQRT(AVG((rs.fraction_high_activity - ra.mean_fha) * 
  (rs.fraction_high_activity - ra.mean_fha))) AS stddev
from round_sim rs, round_avg ra
where rs.round_num = ra.round_num
group by rs.round_num
order by rs.round_num;

select 'metrics on test fraction';

select round_num, round(t_rmse,4) rmse, round(t_r2,4) r2, round(t_spear,4) spearm
from
(
select round_num, avg(train_rmse) tr_rmse, avg(validation_rmse) v_rmse,
  avg(test_rmse) t_rmse, avg(test_r2) t_r2, avg(test_spearman) t_spear
from alde_round r, alde_simulation s, alde_sub_run sr
where r.simulation_id = s.id
  and s.sub_run_id = sr.id
  and sr.run_id = (select max(id) from alde_run)
group by round_num
)
order by round_num;

select 'top variant per round with std';

with 
  round_sim as (
    select round_num, simulation_id, max(assay_score) max_score
    from alde_round_top_variant rv, alde_round r,
      alde_simulation s, alde_sub_run sr
    where rv.round_id = r.id 
      and r.simulation_id = s.id
      and s.sub_run_id = sr.id
      and sr.run_id = (select max(id) from alde_run)
    group by r.id, r.simulation_id
  ),
  round_avg as (
    select round_num, avg(max_score) avg_max_score
    from round_sim
    group by round_num
  )
select rs.round_num, avg(rs.max_score) avg_max_score,
  SQRT(AVG((rs.max_score - ra.avg_max_score) * 
  (rs.max_score - ra.avg_max_score))) AS stddev
from round_sim rs, round_avg ra
where rs.round_num = ra.round_num
group by rs.round_num
order by rs.round_num;

