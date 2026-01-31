SELECT SIMULATION_ID,ROUND_NUM,COUNT(*) FROM (
                select variant_id, sr.strategy_name, round_num, simulation_id,
                       simulation_num, prediction_score, assay_score,
                       num_variants
                from alde_round_top_variant rv, alde_round rnd,
                  alde_simulation s, alde_sub_run sr,
                  alde_run r
                where rv.round_id = rnd.id
                  and rnd.simulation_id = s.id
                  and s.sub_run_id = sr.id
                  and sr.run_id = r.id
                  and r.id =
                    (select max(id)
                    from alde_run r
                    where r.config_id = 'c10'
                        and r.dataset_name = 'ACVRL1'
                        -- and r.name = 'RF_AL_ALL_PRED'
                        and r.end_ts is not null))
GROUP BY SIMULATION_ID,ROUND_NUM
;
