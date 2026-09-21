"""Export a training checkpoint as a standalone model folder that loads directly with
`pipeline("text-to-speech", <folder>)` -- merged weight-norm, no discriminator, with
the tokenizer alongside, so the demo needs nothing from the training setup.
"""
import argparse

from safetensors.torch import load_file
from transformers import AutoTokenizer, VitsModel

from sweep_checkpoints import BASE_MODEL, merge_weight_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="path to a checkpoint-N directory (omit to export the base model)")
    ap.add_argument("--out", required=True, help="output model folder")
    ap.add_argument("--noise_scale", type=float, default=None,
                    help="bake in playback settings (tune_playback.py finds the best)")
    ap.add_argument("--noise_scale_duration", type=float, default=None)
    ap.add_argument("--speaking_rate", type=float, default=None)
    args = ap.parse_args()

    model = VitsModel.from_pretrained(BASE_MODEL)
    unexpected = []
    if args.checkpoint:
        missing, unexpected = model.load_state_dict(
            merge_weight_norm(load_file(f"{args.checkpoint}/model.safetensors")), strict=False
        )
        if missing:
            raise SystemExit(f"checkpoint is missing {len(missing)} weights, refusing to export: {missing[:5]}")

    for name in ["noise_scale", "noise_scale_duration", "speaking_rate"]:
        value = getattr(args, name)
        if value is not None:
            setattr(model.config, name, value)

    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(args.out)
    print(f"[export] {args.checkpoint or BASE_MODEL} -> {args.out} ({len(unexpected)} unused keys ignored)")


if __name__ == "__main__":
    main()
