from argparse import ArgumentParser

a = ArgumentParser()
a.add_argument_group("FlexTrain", "FlexTrain configurations")

a.add_argument(
    "--flextrain",
    default=False,
    action="store_true",
    help="Enable FlexTrain"
)

args = a.parse_args()

print(args)
