-- Delete all runs that are not the latest completed run for each (config_id, name, dataset_name) triplet

.changes ON

begin transaction;

CREATE TEMP TABLE temp_run_ids (id INTEGER);

delete from temp_run_ids;

insert into temp_run_ids
select max(id)
    from alde_run r
    where r.end_ts is not null
    group by r.config_id, r.name, r.dataset_name;
    
delete from alde_last_round_score
where simulation_id in
  (select s.id
  from alde_simulation s, alde_sub_run sr
  where s.sub_run_id = sr.id
    and sr.run_id not in
      (select id from temp_run_ids)
  );


delete from alde_round_top_variant
where round_id in
  (select rnd.id
  from alde_round rnd, alde_simulation s, alde_sub_run sr
  where rnd.simulation_id = s.id
    and s.sub_run_id = sr.id
    and sr.run_id not in
      (select id from temp_run_ids)
  );

delete from alde_round_acquired_variant
where round_id in
  (select rnd.id
  from alde_round rnd, alde_simulation s, alde_sub_run sr
  where rnd.simulation_id = s.id
    and s.sub_run_id = sr.id
    and sr.run_id not in
      (select id from temp_run_ids)
  );

delete from alde_round
where simulation_id in
  (select s.id
  from alde_simulation s, alde_sub_run sr
  where s.sub_run_id = sr.id
    and sr.run_id not in
      (select id from temp_run_ids)
  );

delete from alde_simulation
where sub_run_id in
  (select sr.id
  from alde_sub_run sr
  where sr.run_id not in
      (select id from temp_run_ids)
  );

delete from alde_sub_run
where run_id not in
  (select id from temp_run_ids);

delete from alde_run
where id not in
  (select id from temp_run_ids);
    
