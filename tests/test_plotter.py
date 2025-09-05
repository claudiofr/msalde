from datetime import datetime
from msalde.repository import RepoSessionContext, ALDERepository
from msalde.plotter import Plotter

class DummyConfig:
    """Small config stub just to provide plot titles."""
    roc_title = "Test ROC Curve"
    pr_title = "Test Precision-Recall Curve"

def main():
    # Connect to a test DB (using SQLite memory DB here for simplicity)
    db_url = "sqlite:///:memory:"   # use "sqlite:///./msalde.db" if you want persistence
    session_ctx = RepoSessionContext(db_url)
    repo = ALDERepository(session_ctx)

    # Create a fake run/subrun/simulation
    run = repo.add_run(
        name="TestRun",
        descrip="Testing Plotter",
        num_rounds=1,
        num_variants=10,
        num_selected_variants_first_round=5,
        num_top_acquistion_score_variants_per_round=2,
        num_top_prediction_score_variants_per_round=2,
        batch_size=8,
        test_fraction=0.2,
        random_seed=42,
        max_assay_score=1.0,
        start_ts=datetime.now(),
    )
    sub_run = repo.add_sub_run(
        run_id=run.id,
        learner_name="RidgeRegression",
        learner_parameters="{}",
        first_round_acquisition_strategy_name="Random",
        first_round_acquisition_strategy_parameters="{}",
        acquisition_strategy_name="Greedy",
        acquisition_strategy_parameters="{}",
        start_ts=datetime.now(),
    )
    sim = repo.add_simulation(
        sub_run_id=sub_run.id,
        simulation_num=1,
        start_ts=datetime.now(),
    )

    # Insert some fake ROC + PR data
    for fpr, tpr in [(0.0, 0.0), (0.1, 0.7), (0.2, 0.85), (1.0, 1.0)]:
        repo.add_roc_point(simulation_id=sim.id, fpr=fpr, tpr=tpr)

    for prec, rec in [(1.0, 0.0), (0.9, 0.4), (0.75, 0.7), (0.5, 1.0)]:
        repo.add_pr_point(simulation_id=sim.id, precision=prec, recall=rec)

    # Run the Plotter
    plotter = Plotter(config=DummyConfig(), query_repository=repo)
    plotter.plot_results(simulation_id=sim.id)

if __name__ == "__main__":
    main()
