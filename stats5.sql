select m.run_id rid, substr(embedder_model_name,3,6) model, dataset_name dataset,
  round(tr_rmse,3) tr_rmse, round(tr_r2,3) tr_r2, round(tr_spearm,3) tr_spearm,
  round(v_rmse,3) v_rmse, round(v_r2,3) v_r2, round(v_spearm,3) v_spearm,
  CASE
      WHEN round_num == 1 then 0
      ELSE
          num_variants_first_round + ((round_num-2) *
          (num_top_acq_var_per_round+num_top_pred_var_per_round))
    END AS train_sz,
  mod_param mp,
  substr(r.end_ts,1,10) end_ts
from
(
select run_id, round_num, avg(train_rmse) tr_rmse,
  avg(train_r2) tr_r2, avg(train_spearman) tr_spearm,
  avg(validation_rmse) v_rmse, avg(validation_r2) v_r2,
  avg(validation_spearman) v_spearm,
  avg(test_rmse) t_rmse, avg(test_r2) t_r2, avg(test_spearman) t_spearm,
  substr(sr.model_parameters,17,5) mod_param
from alde_round r, alde_simulation s, alde_sub_run sr
where r.simulation_id = s.id
  and s.sub_run_id = sr.id
  and sr.run_id > 9 --in (12,13,14) -- = (select id from run_id)
group by round_num, run_id
) m, alde_run r
where m.run_id = r.id
  and round_num = 2
  and r.end_ts is not null
order by run_id, round_num;
