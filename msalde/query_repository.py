from sqlalchemy import text
import pandas as pd

class QueryRepository:
    def __init__(self, session):
        self._session = session

    def get_roc_metrics(self, analysis_id: int) -> pd.DataFrame:
        sql = text("""
            SELECT SOURCE_NAME, SCORE_SOURCE, ROC_AUC,
                   NUM_VARIANTS, NUM_POSITIVE_LABELS, NUM_NEGATIVE_LABELS
            FROM roc_metrics
            WHERE analysis_id = :analysis_id
        """)
        res = self._session.execute(sql, {"analysis_id": analysis_id})
        return pd.DataFrame(res.mappings())

    def get_pr_metrics(self, analysis_id: int) -> pd.DataFrame:
        sql = text("""
            SELECT SOURCE_NAME, SCORE_SOURCE, PR_AUC,
                   NUM_VARIANTS, NUM_POSITIVE_LABELS, NUM_NEGATIVE_LABELS
            FROM pr_metrics
            WHERE analysis_id = :analysis_id
        """)
        res = self._session.execute(sql, {"analysis_id": analysis_id})
        return pd.DataFrame(res.mappings())

    def get_mwu_metrics(self, analysis_id: int) -> pd.DataFrame:
        sql = text("""
            SELECT SOURCE_NAME, MWU_STAT, P_VALUE
            FROM mwu_metrics
            WHERE analysis_id = :analysis_id
        """)
        res = self._session.execute(sql, {"analysis_id": analysis_id})
        return pd.DataFrame(res.mappings())
