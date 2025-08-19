from sqlalchemy import (
    Boolean, Column, Integer, Float, String, Date, DateTime, ForeignKey,
    UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ALDERun(Base):
    __tablename__ = 'alde_run'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    descrip = Column(String(500))
    num_rounds = Column(Integer, nullable=False)  # num_rounds
    num_variants = Column(Integer, nullable=False)  # num_variants
    num_variants_first_round = Column(
        Integer, nullable=False)
    num_top_acq_var_per_round = Column(
        Integer, nullable=False)
    num_top_pred_var_per_round = Column(
        Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    test_fraction = Column(Float, nullable=False)
    random_seed = Column(Integer, nullable=False)
    max_assay_score = Column(Float, nullable=False)
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)

    sub_runs = relationship("ALDESubRun", back_populates="run")


class ALDESubRun(Base):
    __tablename__ = 'alde_sub_run'
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('alde_run.id'))
    model_name = Column(String(50), nullable=False)
    model_parameters = Column(String)
    first_round_strategy = Column(String(100), nullable=False)
    first_round_strategy_params = Column(String)
    strategy = Column(String(100), nullable=False)
    strategy_parameters = Column(String)
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)

    run = relationship("ALDERun", back_populates="sub_runs")
    simulations = relationship("ALDESimulation", back_populates="sub_run")


class ALDESimulation(Base):
    __tablename__ = 'alde_simulation'
    id = Column(Integer, primary_key=True)
    sub_run_id = Column(Integer, ForeignKey('alde_sub_run.id'))
    simulation_num = Column(Integer, nullable=False)
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)

    sub_run = relationship("ALDESubRun", back_populates="simulations")
    rounds = relationship("ALDERound", back_populates="simulation")


class ALDERound(Base):
    __tablename__ = 'alde_round'
    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey('alde_simulation.id'))
    round_num = Column(Integer, nullable=False)
    rmse = Column(Float)
    r2 = Column(Float)
    spearman = Column(Float)
    top_n_mean = Column(Float)
    best_variant_id = Column(Integer)
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)

    simulation = relationship("ALDESimulation", back_populates="rounds")
    round_variants = relationship("ALDERoundVariant", back_populates="round")


class ALDERoundVariant(Base):
    __tablename__ = 'alde_round_variant'
    id = Column(Integer, primary_key=True)
    round_id = Column(Integer, ForeignKey('alde_round.id'))
    variant_id = Column(Integer, nullable=False)
    variant_name = Column(String(100))
    assay_score = Column(Float)
    prediction_score = Column(Float)
    acquisition_score = Column(Float)
    top_acquisition_score = Column(Boolean)
    top_prediction_score = Column(Boolean)
    insert_ts = Column(DateTime)

    __table_args__ = (
        UniqueConstraint('round_id', 'variant_id', name='uix_round_variant'),
    )

    round = relationship("ALDERound", back_populates="round_variants")

