import csv

def load_metadata(csv_path: str):

    examples = []

    with open(csv_path, "r") as f:

        reader = csv.DictReader(f)

        for row in reader:

            examples.append(
                (
                    row["audio_path"],
                    row["transcript"]
                )
            )

    return examples