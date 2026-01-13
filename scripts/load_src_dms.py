import context
from msalde.container import ALDEContainer

container = ALDEContainer()

var_loader = container.variant_ref_loader

variant_assay, input_df = var_loader.load_src_dataset()

repo = container.variant_repository
repo.add_variant_assay(variant_assay)
repo.add_variant_assays_bulk(input_df)
