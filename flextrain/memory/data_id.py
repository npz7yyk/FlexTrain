from enum import Enum

from flextrain.utils.distributed import get_rank


class FlexTrainDataTypes(Enum):
    """
    Enum for the data types used in FlexTrain.
    """
    CKPT = 0
    PARA = 1
    GRAD = 2
    OPTS = 3


class FlexTrainDataID:

    def __init__(
        self,
        layer_index: int,
        data_type: FlexTrainDataTypes
    ):
        self.layer_index = layer_index
        self.data_type = data_type

    def __str__(self):
        return (
            f"rank{get_rank()}_"
            f"layer{self.layer_index}_"
            f"{self.data_type.name}.swp"
        )
