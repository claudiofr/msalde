from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from functools import singledispatchmethod
import pandas as pd

from .var_dbmodel import (
    VariantAssay,
    VariantAssayActivity,
    Base
)


class RepoSessionContext:

    def init_db(self):
        # Create all tables
        Base.metadata.create_all(self._engine)
        # create views

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


class VariantRepository:

    def __init__(self, session_context: RepoSessionContext):
        self._engine = session_context.engine

    @singledispatchmethod
    def add_variant_assay(self, arg, *args):
        raise NotImplementedError("Unsupported type")

    @add_variant_assay.register
    def _(
        self,
        variant_assay: VariantAssay,
    ) -> VariantAssay:
        if variant_assay.insert_ts is None:
            variant_assay.insert_ts = datetime.now()
        session = sessionmaker(bind=self._engine)
        with session() as session:
            session.add(variant_assay)
            session.commit()
            session.refresh(variant_assay)
            return variant_assay

    @add_variant_assay.register
    def _(
        self,
        assay_source: str,
        gene_symbol: str,
        protein_symbol: str,
        description: str,
        insert_ts: datetime = datetime.now(),
    ) -> VariantAssay:
        session = sessionmaker(bind=self._engine)
        with session() as session:
            variant_assay = VariantAssayActivity(
                assay_source=assay_source,
                gene_symbol=gene_symbol,
                protein_symbol=protein_symbol,
                description=description,
                insert_ts=insert_ts,
            )
            session.add(variant_assay)
            session.commit()
            session.refresh(variant_assay)
            return variant_assay


    def add_variant_assay_activity(
        self,
        assay_source: str,
        protein_symbol: str,
        assay_type: str,
        assay_subtype: str,
        variant_id: int,
        position: int,
        ref_aa: str,
        var_aa: str,
        assay_score: float,
        alt_assay_score1: float,
        alt_assay_score2: float,
        class_label: int,
        alt_class_label1: int,
        alt_class_label2: int,
        insert_ts: datetime = datetime.now(),
    ) -> VariantAssayActivity:
        session = sessionmaker(bind=self._engine)
        with session() as session:
            variant_assay = VariantAssayActivity(
                assay_source=assay_source,
                protein_symbol=protein_symbol,
                assay_type=assay_type,
                assay_subtype=assay_subtype,
                variant_id=variant_id,
                position=position,
                ref_aa=ref_aa,
                var_aa=var_aa,
                assay_score=assay_score,
                alt_assay_score1=alt_assay_score1,
                alt_assay_score2=alt_assay_score2,
                class_label=class_label,
                alt_class_label1=alt_class_label1,
                alt_class_label2=alt_class_label2,
                insert_ts=insert_ts,
            )
            session.add(variant_assay)
            session.commit()
            session.refresh(variant_assay)
            return variant_assay

    def _df_nvl_column(self, row, column_name: str):
        return row[column_name] if column_name in row.index else None

    def add_variant_assays_bulk(self, variant_assays: pd.DataFrame,
                                insert_ts: datetime = datetime.now()):
        session = sessionmaker(bind=self._engine)
        with session() as session:
            for _, row in variant_assays.iterrows():
                variant_assay = VariantAssayActivity(
                    assay_source=row['assay_source'],
                    protein_symbol=row['protein_symbol'],
                    assay_type=self._df_nvl_column(row, "assay_type"),
                    assay_subtype=self._df_nvl_column(row, "assay_subtype"),
                    variant_id=row["variant_id"],
                    position=self._df_nvl_column(row, "position"),
                    ref_aa=self._df_nvl_column(row, "ref_aa"),
                    var_aa=self._df_nvl_column(row, "var_aa"),
                    assay_score=row["assay_score"],
                    alt_assay_score1=self._df_nvl_column(row, "alt_assay_score1"),
                    alt_assay_score2=self._df_nvl_column(row, "alt_assay_score2"),
                    class_label=self._df_nvl_column(row, "class_label"),
                    alt_class_label1=self._df_nvl_column(row, "alt_class_label1"),
                    alt_class_label2=self._df_nvl_column(row, "alt_class_label2"),
                    insert_ts=insert_ts,
                )
                session.add(variant_assay)
            session.commit()

    def get_variant_assay(self, assay_source: str,
                          assay_type: str = None, 
                          gene_symbol: str = None,
                          protein_symbol: str = None,
                          assay_subtype: str = None) -> pd.DataFrame:

        session = sessionmaker(bind=self._engine)
        with session() as session:
            sql = text("""
            select
                va.assay_source, va.protein_symbol, va.gene_symbol, assay_type,
                assay_subtype, variant_id, position, ref_aa, var_aa,
                assay_score, alt_assay_score1, alt_assay_score2,
                class_label, alt_class_label1, alt_class_label2,
                vaa.insert_ts
            from variant_assay va, variant_assay_activity vaa
            where (va.assay_source = :assay_source)
              and (:assay_type is null or assay_type = :assay_type)
              and (:gene_symbol is null or va.gene_symbol = :gene_symbol)
              and (:protein_symbol is null or va.protein_symbol = :protein_symbol)
              and (:assay_subtype is null or assay_subtype = :assay_subtype)
              and va.assay_source = vaa.assay_source
              and va.protein_symbol = vaa.protein_symbol
            """)
            result = session.execute(
                sql,
                {"assay_source": assay_source,
                 "gene_symbol": gene_symbol,
                 "protein_symbol": protein_symbol,
                 "assay_type": assay_type,
                 "assay_subtype": assay_subtype}
            )
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df

