from sqlalchemy import create_engine
from .dbmodel import Base
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


class RunRepository:

    def __init__(self, session_context: RepoSessionContext):
        self._engine = session_context.engine

    def add(
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

    def end_run(self, run_id, end_ts):
        with self._engine.begin() as session:
            run = session.query(ALDERun).get(run_id)
            run.end_ts = end_ts
            session.flush()
            return run.id
