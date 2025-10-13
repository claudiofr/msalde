from sqlalchemy import (
    Boolean, Column, Integer, Float, String, Date, DateTime, ForeignKey,
    UniqueConstraint, text
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ALDERun(Base):
    __tablename__ = 'alde_run'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    descrip = Column(String(500))
    config_id = Column(String(6), nullable=True)
    data_loader_type = Column(String(20), nullable=False)  # e.g., csv, sqlite
    dataset_name = Column(String(100), nullable=False)
    embedder_type = Column(String(20))  # e.g., plm, onehot
    embedder_model_name = Column(String(100))  # e.g., _8M_UR50D
    embedder_parameters = Column(String)
    log_likelihood_type = Column(String(20))  # e.g., esm, none
    log_likelihood_parameters = Column(String)
    num_simulations = Column(Integer, nullable=False)  # num_simulations
    num_rounds = Column(Integer, nullable=False)  # num_rounds
    num_variants = Column(Integer, nullable=False)  # num_variants
    num_variants_first_round = Column(
        Integer, nullable=False)
    num_top_acq_var_per_round = Column(
        Integer, nullable=False)
    num_top_pred_var_per_round = Column(
        Integer, nullable=False)
    num_top_n_pred_per_round = Column(
        Integer, nullable=True)  # can be null if not used
    batch_size = Column(Integer, nullable=False)
    test_fraction = Column(Float, nullable=False)
    random_seed = Column(Integer, nullable=False)
    max_assay_score = Column(Float, nullable=False)
    binary_score_cutoff = Column(Float)
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)

    sub_runs = relationship("ALDESubRun", back_populates="run")


class ALDESubRun(Base):
    __tablename__ = 'alde_sub_run'
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('alde_run.id'))
    model_type = Column(String(50), nullable=False)
    model_name = Column(String(50), nullable=False)
    model_parameters = Column(String)
    first_round_strategy_type = Column(String(100), nullable=False)
    first_round_strategy_name = Column(String(100), nullable=False)
    first_round_strategy_params = Column(String)
    strategy_type = Column(String(100), nullable=False)
    strategy_name = Column(String(100), nullable=False)
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
    last_round_scores = relationship("ALDELastRoundScore",
                                     back_populates="simulation")


class ALDERound(Base):
    __tablename__ = 'alde_round'
    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey('alde_simulation.id'))
    round_num = Column(Integer, nullable=False)
    train_rmse = Column(Float)
    train_r2 = Column(Float)
    train_spearman = Column(Float)
    validation_rmse = Column(Float)
    validation_r2 = Column(Float)
    validation_spearman = Column(Float)
    test_rmse = Column(Float)
    test_r2 = Column(Float)
    test_spearman = Column(Float)
    spearman = Column(Float)
    top_n_mean = Column(Float)
    best_variant_id = Column(Integer)
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)

    simulation = relationship("ALDESimulation", back_populates="rounds")
    round_variants = relationship("ALDERoundAcquiredVariant",
                                  back_populates="round")
    round_top_variants = relationship("ALDERoundTopVariant",
                                      back_populates="round")


class ALDERoundAcquiredVariant(Base):
    __tablename__ = 'alde_round_acquired_variant'
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


class ALDERoundTopVariant(Base):
    __tablename__ = 'alde_round_top_variant'
    id = Column(Integer, primary_key=True)
    round_id = Column(Integer, ForeignKey('alde_round.id'))
    variant_id = Column(Integer, nullable=False)
    variant_name = Column(String(100))
    assay_score = Column(Float)
    prediction_score = Column(Float)
    insert_ts = Column(DateTime)

    __table_args__ = (
        UniqueConstraint('round_id', 'variant_id', name='uix_round_top_variant'),
    )

    round = relationship("ALDERound", back_populates="round_top_variants")


class ALDELastRoundScore(Base):
    __tablename__ = 'alde_last_round_score'
    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey('alde_simulation.id'))
    variant_id = Column(Integer, nullable=False)
    assay_score = Column(Float)
    prediction_score = Column(Float)
    insert_ts = Column(DateTime)

    __table_args__ = (
        UniqueConstraint('simulation_id', 'variant_id', name='uix_round_top_variant'),
    )

    simulation = relationship("ALDESimulation", back_populates="last_round_scores")
    
    
class RunMetrics(Base):
    __tablename__ = 'run_metrics'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    display_name = Column(String(100))
    run_id = Column(Integer)
    sub_run_id = Column(Integer)
    run_name = Column(String(100))
    run_descrip = Column(String(500))
    config_id = Column(String(6), nullable=True)
    data_loader_type = Column(String(20), nullable=False)  # e.g., csv, sqlite
    dataset_name = Column(String(100), nullable=False)
    embedder_type = Column(String(20), nullable=False)  # e.g., plm, onehot
    embedder_model_name = Column(String(100), nullable=False)  # e.g., esm2_t2_8M_UR50D
    embedder_parameters = Column(String)
    num_simulations = Column(Integer, nullable=False)  # num_simulations
    num_rounds = Column(Integer, nullable=False)  # num_rounds
    num_variants = Column(Integer, nullable=False)  # num_variants
    num_variants_first_round = Column(
        Integer, nullable=False)
    num_top_acq_var_per_round = Column(
        Integer, nullable=False)
    num_top_pred_var_per_round = Column(
        Integer, nullable=False)
    num_top_n_pred_per_round = Column(
        Integer, nullable=True)  # can be null if not used
    batch_size = Column(Integer, nullable=False)
    test_fraction = Column(Float, nullable=False)
    random_seed = Column(Integer, nullable=False)
    max_assay_score = Column(Float, nullable=False)
    binary_score_cutoff = Column(Float)
    model_name = Column(String(50), nullable=False)
    model_parameters = Column(String)
    first_round_strategy = Column(String(100), nullable=False)
    first_round_strategy_params = Column(String)
    strategy = Column(String(100), nullable=False)
    strategy_parameters = Column(String)
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)
    end_round_num = Column(Integer)
    mean_fha = Column(Float)
    mean_fha_rank = Column(Integer)
    mean_activity = Column(Float)
    mean_activity_rank = Column(Integer)
    max_activity = Column(Float)
    max_activity_rank = Column(Integer)
    train_rmse = Column(Float)
    train_rmse_rank = Column(Integer)
    train_r2 = Column(Float)
    train_r2_rank = Column(Integer)
    train_spearm = Column(Float)
    train_spearm_rank = Column(Integer)
    val_rmse = Column(Float)
    val_rmse_rank = Column(Integer)
    val_r2 = Column(Float)
    val_r2_rank = Column(Integer)
    val_spearm = Column(Float)
    val_spearm_rank = Column(Integer)
    total_rank = Column(Float)


class RunMetricsView:
    sql = text("""
            create view if not exists run_metrics_view as
            select
                id,
                end_round_num,
                mean_fha,
                mean_fha_rank,
                mean_activity,
                mean_activity_rank,
                max_activity,
                max_activity_rank,
                train_rmse,
                train_rmse_rank,
                train_r2,
                train_r2_rank,
                train_spearm,
                train_spearm_rank,
                val_rmse,
                val_rmse_rank,
                val_r2,
                val_r2_rank,
                val_spearm,
                val_spearm_rank,
                total_rank,
                name
            from run_metrics
         """)
