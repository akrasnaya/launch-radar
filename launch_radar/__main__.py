"""Run the sample release readout or the local interface."""

import sys

from launch_radar.pipeline import render, run
from launch_radar.sample_data import sample_records


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "dmv":
        from launch_radar.dmv_corpus import render_summary

        print(render_summary())
        return
    if args and args[0] == "serve":
        port = int(args[1]) if len(args) > 1 else 8765
        from launch_radar.server import serve

        serve(port=port)
        return
    records = sample_records()
    _, _, cards = run(records)
    print(render(cards))


if __name__ == "__main__":
    main()
