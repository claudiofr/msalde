from sqlalchemy import (
    Column, Integer, Float, String, Date, DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ALDERun(Base):
    __tablename__ = 'alde_run'
    id = Column(Integer, primary_key=True)
    num_rounds = Column(Integer, nullable=False)  # num_rounds
    num_selected_variants_first_round = Column(
        Integer, nullable=False)  # num_selected_variants_first_round
    num_selected_variants_per_round = Column(
        Integer, nullable=False)  # num_selected_variants_per_round
    batch_size = Column(Integer, nullable=False)
    test_fraction = Column(Float, nullable=False)
    random_seed = Column(Integer, nullable=False)
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)

    sub_runs = relationship("ALDESubRun", back_populates="run")


class ALDESubRun(Base):
    __tablename__ = 'alde_sub_run'
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50), nullable=False)
    model_parameters = Column(String, nullable=False)
    strategy = Column(String(100), nullable=False)
    strategy_parameters = Column(String, nullable=False)
    run_id = Column(Integer, ForeignKey('alde_run.id'))
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)

    run = relationship("ALDERun", back_populates="sub_runs")
    rounds = relationship("ALDERound", back_populates="sub_run")


class ALDERound(Base):
    __tablename__ = 'alde_round'
    id = Column(Integer, primary_key=True)
    sub_run_id = Column(Integer, ForeignKey('alde_sub_run.id'))
    round_num = Column(Integer, nullable=False)
    rmse = Column(Float)
    r2 = Column(Float)
    spearman = Column(Float)
    top_n_mean = Column(Float)
    best_variant_id = Column(Integer)
    max_simulation_score_delta = Column(Float)
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)

    sub_run = relationship("ALDERunStrategy", back_populates="rounds")
    round_variants = relationship("ALDERoundVariant", back_populates="round")


class ALDERoundVariant(Base):
    __tablename__ = 'alde_round_variant'
    id = Column(Integer, primary_key=True)
    variant_name = Column(String(100), nullable=False)
    variant_sequence = Column(String, nullable=False)
    assay_score = Column(Float)
    model_score = Column(Float)
    acquisition_score = Column(Float)
    round_id = Column(Integer, ForeignKey('alde_round.id'))
    insert_ts = Column(DateTime)

    round = relationship("ALDERound", back_populates="round_variants")

