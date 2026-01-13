from sqlalchemy import (
    Boolean, Column, Integer, Float, String, Date, DateTime, ForeignKey,
    UniqueConstraint, text, Text
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class VariantAssayActivity(Base):
    __tablename__ = 'variant_assay_activity'
    id = Column(Integer, primary_key=True)
    assay_source = Column(String(20), nullable=False)
    protein_symbol = Column(String(15), nullable=False)
    assay_type = Column(String(50))
    assay_subtype = Column(String(50))
    variant_id = Column(Integer, nullable=False)
    position = Column(Integer)
    ref_aa = Column(String(1))
    var_aa = Column(String(1))
    assay_score = Column(Float, nullable=False)
    alt_assay_score1 = Column(Float)
    alt_assay_score2 = Column(Float)
    class_label = Column(Integer)
    alt_class_label1 = Column(Integer)
    alt_class_label2 = Column(Integer)
    insert_ts = Column(DateTime)


class VariantAssay(Base):
    __tablename__ = 'variant_assay'
    id = Column(Integer, primary_key=True)
    assay_source = Column(String(20), nullable=False)
    protein_symbol = Column(String(15), nullable=False)
    gene_symbol = Column(String(15), nullable=False)
    description = Column(String(200))
    insert_ts = Column(DateTime)


