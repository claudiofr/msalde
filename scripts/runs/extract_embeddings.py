import context  # noqa: F401 E402

from msalde.container import ALDEContainer


datasets = [
    "MC4R",
    "HXK4",
    "PTEN",
    "SRC",
]
#datasets = ['cas12f2']


# minerva change
def get_alde_container(config_file):
    return ALDEContainer(config_file)


def extract_embeddings(datasets, config_file):
    # parser = create_parser()
    # args = parser.parse_args()
    # configid = args.config_id
    # dataset = args.dataset
    extractor = get_alde_container(config_file).embedding_extractor
    datasets_ = datasets
    extractor.extract_by_dataset_name(datasets_)


if __name__ == "__main__":
    extract_embeddings(datasets, "./config/msaldem.yaml")
