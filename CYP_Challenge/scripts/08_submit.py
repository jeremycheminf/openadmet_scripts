"""Submit to the OpenADMET CYP Challenge HF Space via its Gradio API -- the same
mechanism used for the prior OpenADMET PXR challenge (see
../../PXR/68_auto_submit.py). Confirmed API surface (client.view_api() against
openadmet/cyp-challenge, 2026-09-03):

    predict(username, user_alias, anon_checkbox, participant_name, discord_username,
            email, affiliation, model_tag, paper_checkbox, proprietary_data_checkbox,
            open_code_checkbox, track_select, file_input,
            api_name="/submit_predictions") -> submission_status

    track_select: Literal['Regression Prediction', 'Classification Prediction']

Unlike PXR (one track), CYP needs two separate submission calls, one per track.
Rate limit: as of 2026-09-03, reportedly 24h between submissions (was 4h for PXR --
verify against the actual "Please wait HH:MM:SS" error message if you hit it).

gradio_client isn't installable into this project's conda envs -- use a throwaway
uv venv instead (same trick as scripts/30_fetch_live_leaderboard.py in the main
research repo):
    wsl -e bash -c "mkdir -p /tmp/gradio_probe && cd /tmp/gradio_probe && \
      /home/jeremy/.local/bin/uv venv --python 3.12 .venv && \
      /home/jeremy/.local/bin/uv pip install --python .venv/bin/python gradio_client"
    wsl -e bash -c "cd /mnt/c/.../submission_kit && /tmp/gradio_probe/.venv/bin/python \
      scripts/08_submit.py --activity results/submission_activity.csv \
      --tdi results/submission_tdi.csv --model-tag 'my model description'"

REVIEW THE IDENTITY FIELDS BELOW before running -- this is a real, visible
submission (affects the public leaderboard, rate-limited to one per track per day).
"""

import argparse
from pathlib import Path

from gradio_client import Client, handle_file

# --- Submission identity -- REVIEW before running -----------------------------
HF_USERNAME = "REDACTED_HF_USERNAME"
ALIAS = "jeremy"
ANON = True
FULL_NAME = "REDACTED_NAME"
DISCORD = "jeremycheminf"
EMAIL = "REDACTED_EMAIL"
AFFILIATION = "REDACTED_AFFILIATION"
PAPER = False
PROPRIETARY_DATA = False
OPEN_CODE = True  # this kit is meant to be shared publicly
MODEL_LINK = "https://github.com/jeremycheminf/openadmet_scripts/tree/main/CYP_Challenge"


def submit(client: Client, csv_path: Path, track: str, model_tag: str) -> str:
    status = client.predict(
        HF_USERNAME, ALIAS, ANON, FULL_NAME, DISCORD, EMAIL, AFFILIATION,
        model_tag, PAPER, PROPRIETARY_DATA, OPEN_CODE, track,
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

    model_tag = f"{args.model_tag} — {MODEL_LINK}" if MODEL_LINK else args.model_tag

    if args.dry_run:
        print(f"[DRY RUN] would submit as {ALIAS} ({FULL_NAME}, {AFFILIATION}), tag={model_tag!r}")
        if args.activity:
            print(f"  Regression Prediction <- {args.activity}")
        if args.tdi:
            print(f"  Classification Prediction <- {args.tdi}")
        return

    client = Client("openadmet/cyp-challenge", verbose=False)
    if args.activity:
        status = submit(client, args.activity, "Regression Prediction", model_tag)
        print(f"Regression: {status}")
    if args.tdi:
        status = submit(client, args.tdi, "Classification Prediction", model_tag)
        print(f"Classification: {status}")


if __name__ == "__main__":
    main()
