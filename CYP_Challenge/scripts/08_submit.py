"""Submit to the OpenADMET CYP Challenge HF Space via its Gradio API. Confirmed API
surface (client.view_api() against openadmet/cyp-challenge, 2026-09-03):

    predict(username, user_alias, anon_checkbox, participant_name, discord_username,
            email, affiliation, model_tag, paper_checkbox, proprietary_data_checkbox,
            open_code_checkbox, track_select, file_input,
            api_name="/submit_predictions") -> submission_status

    track_select: Literal['Regression Prediction', 'Classification Prediction']

CYP needs two separate submission calls, one per track. Rate limit: as of
2026-09-03, reportedly 24h between submissions -- verify against the actual
"Please wait HH:MM:SS" error message if you hit it.

gradio_client isn't installable into this project's conda envs -- use a throwaway
uv venv instead:
    wsl -e bash -c "mkdir -p /tmp/gradio_probe && cd /tmp/gradio_probe && \
      /home/jeremy/.local/bin/uv venv --python 3.12 .venv && \
      /home/jeremy/.local/bin/uv pip install --python .venv/bin/python gradio_client"

Identity fields are read from environment variables (never hardcoded here) -- set
them before running, e.g.:
    export CYP_SUBMIT_HF_USERNAME=...
    export CYP_SUBMIT_ALIAS=...
    export CYP_SUBMIT_FULL_NAME=...
    export CYP_SUBMIT_DISCORD=...
    export CYP_SUBMIT_EMAIL=...
    export CYP_SUBMIT_AFFILIATION=...
    export CYP_SUBMIT_MODEL_LINK=...   # optional; if set, submitted bare as the
                                        # "Method Report Link" (open_code=True),
                                        # overriding --model-tag entirely

Then run, e.g.:
    python scripts/08_submit.py --activity results/submission_activity.csv \
      --tdi results/submission_tdi.csv --model-tag 'my model description'

REVIEW THE IDENTITY FIELDS before running -- this is a real, visible submission
(affects the public leaderboard, rate-limited to roughly one per track per day).
"""

import argparse
import os
from pathlib import Path

from gradio_client import Client, handle_file

REQUIRED_ENV_VARS = [
    "CYP_SUBMIT_HF_USERNAME", "CYP_SUBMIT_ALIAS", "CYP_SUBMIT_FULL_NAME",
    "CYP_SUBMIT_DISCORD", "CYP_SUBMIT_EMAIL", "CYP_SUBMIT_AFFILIATION",
]
ANON = True
PAPER = False
PROPRIETARY_DATA = False


def submit(client: Client, csv_path: Path, track: str, model_tag: str, open_code: bool) -> str:
    status = client.predict(
        os.environ["CYP_SUBMIT_HF_USERNAME"], os.environ["CYP_SUBMIT_ALIAS"], ANON,
        os.environ["CYP_SUBMIT_FULL_NAME"], os.environ["CYP_SUBMIT_DISCORD"],
        os.environ["CYP_SUBMIT_EMAIL"], os.environ["CYP_SUBMIT_AFFILIATION"],
        model_tag, PAPER, PROPRIETARY_DATA, open_code, track,
        handle_file(str(csv_path)),
        api_name="/submit_predictions",
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity", type=Path, help="path to submission_activity.csv")
    parser.add_argument("--tdi", type=Path, help="path to submission_tdi.csv")
    parser.add_argument("--model-tag", required=True, help="short description shown on the leaderboard")
    parser.add_argument("--dry-run", action="store_true", help="print what would be submitted, don't call the API")
    args = parser.parse_args()

    if not args.activity and not args.tdi:
        parser.error("pass at least one of --activity / --tdi")

    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing and not args.dry_run:
        parser.error(f"missing required environment variables: {', '.join(missing)} (see this script's docstring)")

    # The CYP Space validates model_tag as a clickable "Method Report Link" when
    # open_code_checkbox is True -- it must be a bare, fetchable URL, not a
    # description with a URL appended (confirmed by trial: "description -- URL"
    # was rejected with "Could not open the Method Report Link"). So: if
    # CYP_SUBMIT_MODEL_LINK is set, submit it bare with open_code=True and drop
    # the free-text description; otherwise submit the plain description with
    # open_code=False.
    model_link = os.environ.get("CYP_SUBMIT_MODEL_LINK", "")
    open_code = bool(model_link)
    model_tag = model_link if model_link else args.model_tag

    if args.dry_run:
        alias = os.environ.get("CYP_SUBMIT_ALIAS", "<CYP_SUBMIT_ALIAS not set>")
        full_name = os.environ.get("CYP_SUBMIT_FULL_NAME", "<CYP_SUBMIT_FULL_NAME not set>")
        affiliation = os.environ.get("CYP_SUBMIT_AFFILIATION", "<CYP_SUBMIT_AFFILIATION not set>")
        print(f"[DRY RUN] would submit as {alias} ({full_name}, {affiliation}), "
              f"tag={model_tag!r}, open_code={open_code}")
        if missing:
            print(f"  (note: {', '.join(missing)} not currently set in the environment)")
        if args.activity:
            print(f"  Regression Prediction <- {args.activity}")
        if args.tdi:
            print(f"  Classification Prediction <- {args.tdi}")
        return

    client = Client("openadmet/cyp-challenge", verbose=False)
    if args.activity:
        status = submit(client, args.activity, "Regression Prediction", model_tag, open_code)
        print(f"Regression: {status}")
    if args.tdi:
        status = submit(client, args.tdi, "Classification Prediction", model_tag, open_code)
        print(f"Classification: {status}")


if __name__ == "__main__":
    main()
