                insert into alde_round_top_variant(id, variant_id,
                       valid_prediction_score, valid_round_id)
                select tv.id, tv.variant_id,
                    (select tvi.valid_prediction_score
                    from alde_round rndi, alde_round_top_variant tvi
                    where rnd.simulation_id = rndi.simulation_id
                        and rndi.id = tvi.round_id
                        and tvi.variant_id = tv.variant_id
                        and rndi.round_num < rnd.round_num
                        and tvi.valid_prediction_score is not null
                    order by rndi.round_num desc
                    limit 1) valid_prediction_score,
                    (select rndi.id
                    from alde_round rndi, alde_round_top_variant tvi
                    where rnd.simulation_id = rndi.simulation_id
                        and rndi.id = tvi.round_id
                        and tvi.variant_id = tv.variant_id
                        and rndi.round_num < rnd.round_num
                        and tvi.valid_prediction_score is not null
                    order by rndi.round_num desc
                    limit 1) valid_round_id
                from alde_round rnd, alde_round_top_variant tv
                where rnd.simulation_id = 250
                    and rnd.id = tv.round_id
                    and tv.valid_prediction_score is null
                    and tv.variant_id not in
                        (select av1.variant_id
                        from alde_round rnd1, alde_round_acquired_variant av1
                        where rnd1.id = av1.round_id
                        and rnd1.simulation_id = rnd.simulation_id
                        and rnd1.round_num = 1)
                on conflict(id)
                do update set valid_prediction_score = excluded.valid_prediction_score,
                             valid_round_id = excluded.valid_round_id;
