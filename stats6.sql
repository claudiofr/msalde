select r.*
from alde_sub_run sr, alde_simulation s, alde_round r
where sr.run_id in (select max(id) from alde_run)
  and sr.id = s.sub_run_id
  and r.simulation_id = s.id
order by r.id
