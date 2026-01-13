from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
import pandas as pd

from .dbmodel import Dataset, ALDERun

from .repository import RepoSessionContext


class ALDEQueryRepository:

    def __init__(self, session_context: RepoSessionContext):
        self._engine = session_context.engine

    def get_run_by_config_dataset_run(
        self, config_id: int, dataset_name: str, run_name: str
        ) -> ALDERun:
        session = sessionmaker(bind=self._engine)
        with session() as session:
            sql = text("""
            select *
            from alde_run r
            where r.config_id = :config_id
                and r.dataset_name = :dataset_name
                and r.name = :run_name
                and r.end_ts is not null
            order by r.id desc
            limit 1
            """)

            # Execute the query with a parameter
            result = session.execute(
                sql,
                {"config_id": config_id,
                 "dataset_name": dataset_name,
                 "run_name": run_name})
            row = result.fetchone()
            if row:
                run = ALDERun(**row._mapping)
                return run
            else:
                return None

    def get_last_round_scores_by_config_dataset_run(
        self, config_id: int, dataset_name: str, run_name: str
        ) -> pd.DataFrame:

        run = self.get_run_by_config_dataset_run(
            config_id, dataset_name, run_name)
        if not run:
            return pd.DataFrame()
        session = sessionmaker(bind=self._engine)
        with session() as session:
            if run.save_all_predictions:
                sql = text("""
                with
                    predictions as (
                    select variant_id, assay_score,
                                avg(prediction_score) prediction_score, num_variants
                    from alde_simulation s,
                        alde_round rnd,
                        alde_round_top_variant lrs,
                        alde_sub_run sr,
                        alde_run r
                    where s.id = rnd.simulation_id
                        and rnd.id = lrs.round_id
                        and s.sub_run_id = sr.id
                        and sr.run_id = r.id
                        and rnd.round_num = r.num_rounds
                        and r.id = :run_id
                    group by variant_id, lrs.assay_score, num_variants
                    ),
                    mean_std as (
                    select avg(prediction_score) avg_prediction_score,
                        stddev(prediction_score) std_prediction_score
                    from predictions
                    )
                select variant_id, assay_score, prediction_score, num_variants,
                    round((prediction_score - avg_prediction_score)/
                        std_prediction_score, 3) z_score
                from predictions p, mean_std ms
                order by prediction_score desc
                """)
            else:
                sql = text("""
                with
                    predictions as (
                    select variant_id, assay_score,
                                avg(prediction_score) prediction_score, num_variants
                    from alde_simulation s,
                        alde_last_round_score lrs,
                        alde_sub_run sr,
                        alde_run r
                    where s.id = lrs.simulation_id
                        and s.sub_run_id = sr.id
                        and sr.run_id = r.id
                        and r.id = :run_id
                    group by variant_id, lrs.assay_score, num_variants
                    ),
                    mean_std as (
                    select avg(prediction_score) avg_prediction_score,
                        stddev(prediction_score) std_prediction_score
                    from predictions
                    )
                select variant_id, assay_score, prediction_score, num_variants,
                    round((prediction_score - avg_prediction_score)/
                        std_prediction_score, 3) z_score
                from predictions p, mean_std ms
                order by prediction_score desc
                """)

            # Execute the query with a parameter
            result = session.execute(
                sql,
                {"run_id": run.id})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            # Process the results
            return df

    def get_mean_activity_of_top_variants_by_round(
        self, config_id: int, dataset_name: str, run_name: str,
        strategy_name: str = None
        ) -> pd.DataFrame:

        session = sessionmaker(bind=self._engine)
        with session() as session:
            sql = text("""
            with round_sim as (
                select sr.strategy_name, round_num, simulation_id,
                       avg(assay_score) mean_score, num_variants
                from alde_round_top_variant rv, alde_round rnd,
                  alde_simulation s, alde_sub_run sr,
                  alde_run r
                where rv.round_id = rnd.id
                  and rnd.simulation_id = s.id
                  and s.sub_run_id = sr.id
                  and sr.run_id = r.id
                  and (:strategy_name is null
                     or strategy_name = :strategy_name)
                  and r.id =
                    (select max(id)
                    from alde_run r
                    where r.config_id = :config_id
                        and r.dataset_name = :dataset_name
                        and r.name = :run_name
                        and r.end_ts is not null)
                group by sr.strategy_name, rnd.id, rnd.simulation_id,
                       num_variants
            )
            select rs.strategy_name, rs.round_num,
                       avg(rs.mean_score) mean_score,
                       stddev(rs.mean_score) stddev, num_variants
            from round_sim rs
            group by rs.strategy_name, rs.round_num, rs.num_variants
            order by rs.strategy_name, rs.round_num
            """)

            # Execute the query with a parameter
            result = session.execute(
                sql,
                {"config_id": config_id,
                 "dataset_name": dataset_name,
                 "run_name": run_name,
                 "strategy_name": strategy_name})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            # Process the results
            return df

    def get_fractional_high_activity_variants_by_round(
        self, config_id: int, dataset_name: str, run_name: str
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
                    assay_score, num_variants
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
                        where r.config_id = :config_id
                            and r.dataset_name = :dataset_name
                            and r.name = :run_name
                            and r.end_ts is not null)
            ),
            round_sim as (
                select round_num, simulation_id,
                    avg(high_activity) fraction_high_activity
                from high_activity ha
                group by ha.round_num, ha.simulation_id
            )
            select rs.round_num, avg(rs.fraction_high_activity) mean_fha,
                stddev(rs.fraction_high_activity) stddev, num_variants
            from round_sim rs
            group by rs.round_num, num_variants
            order by rs.round_num;
            """)

            # Execute the query with a parameter
            result = session.execute(
                sql,
                {"config_id": config_id,
                 "dataset_name": dataset_name,
                 "run_name": run_name})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            # Process the results
            return df

    def get_gene_symbol_for_dataset(
            self, dataset_name: str) -> str:
        dataset = self.get_dataset_by_name(dataset_name)
        if dataset and dataset.gene_symbol:
            return dataset.gene_symbol
        else:
            return dataset_name

    def get_dataset_by_name(self, dataset_name: str) -> Dataset:
        session = sessionmaker(bind=self._engine)
        with session() as session:
            dataset = session.get(Dataset, dataset_name)
            return dataset
