import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str
    )

    args = parser.parse_args()

    run(model=args.model)