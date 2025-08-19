from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .model import PerformanceMetrics
from .dbmodel import ALDERound, ALDERoundVariant, ALDESimulation, ALDESubRun, Base
from .dbmodel import ALDERun


class RepoSessionContext:

    def init_db(self):
        Base.metadata.create_all(self._engine)

    def __init__(self, db_url: str):
        self._db_url = db_url
        self._engine = create_engine(db_url)
        self.init_db()

    @property
    def db_url(self):
        return self._db_url

    @property
    def engine(self):
        return self._engine


class ALDERepository:

    def __init__(self, session_context: RepoSessionContext):
        self._engine = session_context.engine

    def add_run(
        self,
        name: str,
        descrip: str,
        num_rounds: int,
        num_variants: int,
        num_selected_variants_first_round: int,
        num_top_acquistion_score_variants_per_round: int,
        num_top_prediction_score_variants_per_round: int,
        batch_size: int,
        test_fraction: float,
        random_seed: int,
        max_assay_score: float,
        start_ts: datetime = datetime.now(),
    ) -> ALDERun:
        session = sessionmaker(bind=self._engine)
        with session() as session:
            run = ALDERun(
                name=name,
                descrip=descrip,
                num_rounds=num_rounds,
                num_variants=num_variants,
                num_variants_first_round=
                num_selected_variants_first_round,
                num_top_acq_var_per_round=
                num_top_acquistion_score_variants_per_round,
                num_top_pred_var_per_round=
                num_top_prediction_score_variants_per_round,
                batch_size=batch_size,
                test_fraction=test_fraction,
                random_seed=random_seed,
                max_assay_score=max_assay_score,
                start_ts=start_ts,
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return run

    def end_run(self, id, end_ts):
        session = sessionmaker(bind=self._engine)
        with session() as session:
            run = session.query(ALDERun).get(id)
            run.end_ts = end_ts
            session.commit()

    def add_sub_run(
        self,
        run_id,
        learner_name,
        learner_parameters,
        first_round_acquisition_strategy_name,
        first_round_acquisition_strategy_parameters,
        acquisition_strategy_name,
        acquisition_strategy_parameters,
        start_ts,
    ) -> ALDESubRun:
        session = sessionmaker(bind=self._engine)
        with session() as session:
            sub_run = ALDESubRun(
                model_name=learner_name,
                model_parameters=learner_parameters,
                first_round_strategy=first_round_acquisition_strategy_name,
                first_round_strategy_params=
                first_round_acquisition_strategy_parameters,
                strategy=acquisition_strategy_name,
                strategy_parameters=acquisition_strategy_parameters,
                run_id=run_id,
                start_ts=start_ts,
            )
            session.add(sub_run)
            session.commit()
            session.refresh(sub_run)
            return sub_run

    def end_sub_run(self, id, end_ts):
        session = sessionmaker(bind=self._engine)
        with session() as session:
            sub_run = session.query(ALDESubRun).get(id)
            sub_run.end_ts = end_ts
            session.commit()

    def add_simulation(
        self,
        sub_run_id,
        simulation_num,
        start_ts,
    ) -> ALDESubRun:
        session = sessionmaker(bind=self._engine)
        with session() as session:
            simulation = ALDESimulation(
                sub_run_id=sub_run_id,
                simulation_num=simulation_num,
                start_ts=start_ts,
            )
            session.add(simulation)
            session.commit()
            session.refresh(simulation)
            return simulation

    def end_simulation(self, id, end_ts):
        session = sessionmaker(bind=self._engine)
        with session() as session:
            sub_run = session.query(ALDESimulation).get(id)
            sub_run.end_ts = end_ts
            session.commit()

    def add_round(self, simulation_id, round_num, start_ts) -> ALDERound:
        """
        Add a new round to the sub-run.
        """
        session = sessionmaker(bind=self._engine)
        with session() as session:
            # Create a new round instance
            round = ALDERound(
                simulation_id=simulation_id,
                round_num=round_num,
                start_ts=start_ts,
            )
            session.add(round)
            session.commit()
            session.refresh(round)
            return round

    def end_round(self, round_id, performance_metrics: PerformanceMetrics,
                  best_variant_id: int, end_ts: datetime
                  ) -> ALDERound:
        """
        End a round by updating its performance metrics.
        """
        session = sessionmaker(bind=self._engine)
        with session() as session:
            round = session.query(ALDERound).get(round_id)
            round.rmse = performance_metrics.rmse
            round.r2 = performance_metrics.r2
            round.spearman = performance_metrics.spearman
            round.top_n_mean = performance_metrics.top_n_mean
            round.best_variant_id = best_variant_id
            round.end_ts = end_ts
            session.commit()
            return round

    def add_round_variant(
        self,
        round_id: int,
        variant_id: int,
        variant_name=None,
        assay_score=None,
        acquisition_score=None,
        prediction_score=None,
        top_acquisition_score=False,
        top_prediction_score=False,
        insert_ts=None,
    ) -> ALDERoundVariant:
        session = sessionmaker(bind=self._engine)
        with session() as session:
            round_variant = ALDERoundVariant(
                round_id=round_id,
                variant_id=variant_id,
                variant_name=variant_name,
                assay_score=assay_score,
                acquisition_score=acquisition_score,
                prediction_score=prediction_score,
                top_acquisition_score=top_acquisition_score,
                top_prediction_score=top_prediction_score,
                insert_ts=insert_ts,
            )
            session.add(round_variant)
            session.commit()
            session.refresh(round_variant)
            return round_variant

