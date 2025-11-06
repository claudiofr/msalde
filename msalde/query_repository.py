from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
import pandas as pd

from .dbutil import DbExtensionCreator, DbViewCreator

from .model import PerformanceMetrics
from .dbmodel import (
    ALDERound, ALDERoundAcquiredVariant, ALDESimulation,
    ALDESubRun, Base, ALDERoundTopVariant
)
from .dbmodel import ALDERun
from .repository import RepoSessionContext


class ALDEQueryRepository:

    def __init__(self, session_context: RepoSessionContext):
        self._engine = session_context.engine

    def get_top_variants_by_dataset_learner(
        self, dataset_name: str, learner_name: str
        ) -> pd.DataFrame:

        session = sessionmaker(bind=self._engine)
        with session() as session:
            sql = text("""
            select variant_id, assay_score,
                       avg(prediction_score) prediction_score, num_variants
            from alde_simulation s,
                alde_last_round_score lrs,
                alde_sub_run sr,
                alde_run r
            where s.id = lrs.simulation_id
                and s.sub_run_id = sr.id
                and sr.run_id = r.id
                and s.sub_run_id = (
                select max(sr.id)
                from alde_run r, alde_sub_run sr
                where r.dataset_name = :dataset_name
                    and r.id = sr.run_id
                    and sr.model_name = :learner_name
                    and r.end_ts is not null
            )
            group by variant_id, lrs.assay_score
            order by prediction_score desc
            """)

            # Execute the query with a parameter
            result = session.execute(
                sql,
                {"dataset_name": dataset_name,
                 "learner_name": learner_name})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            # Process the results
            return df

    def get_mean_activity_of_top_variants_by_round(
        self, dataset_name: str, learner_name: str
        ) -> pd.DataFrame:

        session = sessionmaker(bind=self._engine)
        with session() as session:
            sql = text("""
            with round_sim as (
                select round_num, simulation_id, avg(assay_score) mean_score
                from alde_round_top_variant rv, alde_round r,
                  alde_simulation s
                where rv.round_id = r.id
                  and r.simulation_id = s.id
                  and s.sub_run_id = 
                  (select max(sr.id)
                  from alde_run r, alde_sub_run sr
                  where r.dataset_name = :dataset_name
                    and sr.model_name = :learner_name
                    and r.id = sr.run_id
                    and r.end_ts is not null)
                group by r.id, r.simulation_id
            )
            select rs.round_num, avg(rs.mean_score) mean_score,
            stddev(rs.mean_score) stddev
            from round_sim rs
            group by rs.round_num
            order by rs.round_num
            """)

            # Execute the query with a parameter
            result = session.execute(
                sql,
                {"dataset_name": dataset_name,
                 "learner_name": learner_name})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            # Process the results
            return df

    def get_fractional_high_activity_variants_by_round(
        self, dataset_name: str, learner_name: str
        ) -> pd.DataFrame:

        session = sessionmaker(bind=self._engine)
        with session() as session:
            sql = text("""
            with high_activity as (
                select round_num, simulation_id,
                    CASE
                        WHEN assay_score > r.wt_assay_score THEN 1
                        ELSE 0
                    END AS high_activity,
                    assay_score
                from alde_round_top_variant rv, alde_round rnd,
                    alde_simulation s, alde_sub_run sr,
                    alde_run r
                where rv.round_id = rnd.id 
                    and rnd.simulation_id = s.id
                    and s.sub_run_id = sr.id
                    and sr.run_id = r.id
                    and s.sub_run_id = 
                        (select max(sr.id)
                        from alde_run r, alde_sub_run sr
                        where r.dataset_name = :dataset_name
                            and sr.model_name = :learner_name
                            and r.id = sr.run_id
                            and r.end_ts is not null)
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
            """)

            # Execute the query with a parameter
            result = session.execute(
                sql,
                {"dataset_name": dataset_name,
                 "learner_name": learner_name})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            # Process the results
            return df

