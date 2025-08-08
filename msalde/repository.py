from sqlalchemy import create_engine
from .dbmodel import ALDESubRun, Base
from .dbmodel import ALDERun

class RepoSessionContext:
    def __init__(self, db_url: str):
        self._db_url = db_url
        self._engine = create_engine(db_url)

    @property
    def db_url(self):
        return self._db_url

    @property
    def engine(self):
        return self._engine

    def init_db(self):
        Base.metadata.create_all(self._engine)


class ALDERepository:

    def __init__(self, session_context: RepoSessionContext):
        self._engine = session_context.engine

    def add_run(
        self,
        num_rounds,
        num_selected_variants_first_round,
        num_selected_variants_per_round,
        batch_size,
        test_fraction,
        random_seed,
        start_ts,
        end_ts,
    ):
        with self._engine.begin() as session:
            run = ALDERun(
                num_rounds=num_rounds,
                num_selected_variants_first_round=
                num_selected_variants_first_round,
                num_selected_variants_per_round=
                num_selected_variants_per_round,
                batch_size=batch_size,
                test_fraction=test_fraction,
                random_seed=random_seed,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            session.add(run)
            session.flush()
            return run

    def end_run(self, id, end_ts):
        with self._engine.begin() as session:
            run = session.query(ALDERun).get(id)
            run.end_ts = end_ts
            session.flush()
            return run.id

    def add_sub_run(
        self,
        run_id,
        learner_name,
        learner_parameters,
        acquisition_strategy_name,
        acquisition_strategy_parameters,
        start_ts,
    ):
        with self._engine.begin() as session:
            sub_run = ALDESubRun(
                model_name=learner_name,
                model_parameters=learner_parameters,
                strategy=acquisition_strategy_name,
                strategy_parameters=acquisition_strategy_parameters,
                run_id=run_id,
                start_ts=start_ts,
            )
            session.add(sub_run)
            session.flush()
            return sub_run

    def end_sub_run(self, id, end_ts):
        with self._engine.begin() as session:
            sub_run = session.query(ALDESubRun).get(id)
            sub_run.end_ts = end_ts
            session.flush()
            return sub_run.id

