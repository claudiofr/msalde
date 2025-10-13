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
                       avg(prediction_score) prediction_score
            from alde_simulation s,
                alde_last_round_score lrs
            where s.id = lrs.simulation_id
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


