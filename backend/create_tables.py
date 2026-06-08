from app.core.database import Base, engine
from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.models.experiment_metric import ExperimentMetric
from app.models.experiment_parameter import ExperimentParameter
from app.models.project import Project


def main() -> None:
    Base.metadata.create_all(bind=engine)
    print("Таблицы успешно созданы или уже существуют.")


if __name__ == "__main__":
    main()
