.shell rm junk.lis
.output junk.lis


create table run_id(id integer primary key);
delete from run_id;

insert into run_id(id) values( (select max(id) from run_id) );
-- update run_id set id = 5;

.header off

select 'mean activity of top n mutants per round with std';
select '  ';

.header on

with 
  round_sim as (
    select round_num, simulation_id, avg(assay_score) mean_score
    from alde_round_top_variant rv, alde_round r,
      alde_simulation s, alde_sub_run sr
    where rv.round_id = r.id
      and r.simulation_id = s.id
      and s.sub_run_id = sr.id
      and sr.run_id = (select id from run_id)
    group by r.id, r.simulation_id
  )
select rs.round_num, avg(rs.mean_score) mean_score,
  stddev(rs.mean_score) stddev
from round_sim rs
group by rs.round_num
order by rs.round_num;



.header off
select ' ';
select 'fraction high activity mutants by round with std';
select '  ';

.header on

with 
  high_activity as (
    select round_num, simulation_id,
    CASE
      WHEN assay_score > 1 THEN 1
      -- when assay_score > 0.657676441 then 1
      ELSE 0
    END AS high_activity,
    assay_score
    from alde_round_top_variant rv, alde_round r,
      alde_simulation s, alde_sub_run sr
    where rv.round_id = r.id 
      and r.simulation_id = s.id
      and s.sub_run_id = sr.id
      and sr.run_id = (select id from run_id)
  ),
  round_sim as (
    select round_num, simulation_id,
      avg(high_activity) fraction_high_activity
    from high_activity ha
    group by ha.round_num, ha.simulation_id
  )
select rs.round_num, avg(rs.fraction_high_activity) mean_fha,
  stddev(rs.fraction_high_activity) stddev
from round_sim rs
group by rs.round_num
order by rs.round_num;

.header off
select ' ';
select 'standard performance metrics';
select '  ';

.header on

select round_num, round(t_rmse,3) t_rmse, round(t_r2,3) t_r2, round(t_spearm,3) t_spearm,
  round(tr_rmse,3) tr_rmse, round(tr_r2,3) tr_r2, round(tr_spearm,3) tr_spearm,
  round(v_rmse,3) v_rmse, round(v_r2,3) v_r2, round(v_spearm,3) v_spearm
from
(
select round_num, simulation_id, avg(train_rmse) tr_rmse,
  avg(train_r2) tr_r2, avg(train_spearman) tr_spearm,
  avg(validation_rmse) v_rmse, avg(validation_r2) v_r2,
  avg(validation_spearman) v_spearm,
  avg(test_rmse) t_rmse, avg(test_r2) t_r2, avg(test_spearman) t_spearm
from alde_round r, alde_simulation s, alde_sub_run sr
where r.simulation_id = s.id
  and s.sub_run_id = sr.id
  and sr.run_id = (select id from run_id)
group by round_num
)
order by round_num;


.header off
select ' ';
select 'top variant per round with std';
select ' ';
.header on

with 
  round_sim as (
    select round_num, simulation_id, max(assay_score) max_score
    from alde_round_top_variant rv, alde_round r,
      alde_simulation s, alde_sub_run sr
    where rv.round_id = r.id 
      and r.simulation_id = s.id
      and s.sub_run_id = sr.id
      and sr.run_id = (select id from run_id)
    group by r.id, r.simulation_id
  )
select rs.round_num, avg(rs.max_score) avg_max_score,
  stddev(rs.max_score) stddev
from round_sim rs
group by rs.round_num
order by rs.round_num;


.header off
select ' ';
select 'mutant counts per round';
select ' ';
.header on

with
  round_sim as (
    select round_num, simulation_id, count(1) cnt,
      sum(top_acquisition_score) cnt_acquired,
      sum(top_prediction_score) cnt_predicted
    from alde_round_acquired_variant rv, alde_round r,
      alde_simulation s, alde_sub_run sr
    where rv.round_id = r.id 
      and r.simulation_id = s.id
      and s.sub_run_id = sr.id
      and sr.run_id = (select id from run_id)
    group by r.id, r.simulation_id
  )
select rs.round_num, avg(rs.cnt) avg_count,
  avg(cnt_acquired) avg_cnt_acquired,
  avg(cnt_predicted) avg_cnt_predicted
from round_sim rs
group by rs.round_num
order by rs.round_num;


select round_num, 
  round(tr_rmse,3) tr_rmse, round(tr_r2,3) tr_r2, round(tr_spearm,3) tr_spearm,
  round(v_rmse,3) v_rmse, round(v_r2,3) v_r2, round(v_spearm,3) v_spearm,
  CASE
      WHEN round_num == 1 then 0
      ELSE
          num_variants_first_round + ((round_num-2) *
          (num_top_acq_var_per_round+num_top_pred_var_per_round))
    END AS train_size
from
(
select run_id, round_num, avg(train_rmse) tr_rmse,
  avg(train_r2) tr_r2, avg(train_spearman) tr_spearm,
  avg(validation_rmse) v_rmse, avg(validation_r2) v_r2,
  avg(validation_spearman) v_spearm,
  avg(test_rmse) t_rmse, avg(test_r2) t_r2, avg(test_spearman) t_spearm
from alde_round r, alde_simulation s, alde_sub_run sr
where r.simulation_id = s.id
  and s.sub_run_id = sr.id
  and sr.run_id = (select id from run_id)
group by round_num, run_id
) m, alde_run r
where m.run_id = r.id
order by round_num;


.output stdout