"""Draw a saved figure recipe (``.qfig``) from the current output files.

    qali --pipeline figure Rb_grotrian.qfig                      # writes Rb_grotrian.pdf next to it
    qali --pipeline figure fig.qfig --out fig.png --species Cs   # another format or species
    qali --pipeline figure fig.qfig --script fig.py              # also a standalone Python script
    qali --pipeline figure fig.qfig --input lifetimes=hot        # a run saved under a name (Save as), or a file

Recipes are made in the GUI's Figures tab (Save...). Rerun this after a stage changes its output and the
figure follows.
"""

from __future__ import annotations

import argparse
import os
import sys

__version__ = "1.0"


def build_parser() -> argparse.ArgumentParser:
    from qali.figures.spec import SAVE_FORMATS

    p = argparse.ArgumentParser(prog="qali --pipeline figure", description="Draw a saved figure recipe (.qfig).")
    p.add_argument("recipe", help="figure recipe saved from the Figures tab (.qfig)")
    p.add_argument("--out", default=None,
                   help="output file; the extension sets the format (" + ", ".join(SAVE_FORMATS) +
                        "). Default: the recipe's name with .pdf")
    p.add_argument("--species", default=None, help="draw this species instead of the recipe's")
    p.add_argument("--data-dir", default=None, help="folder with the stage outputs (default: the workspace's)")
    p.add_argument("--script", default=None, help="also write a standalone Python script that draws the figure")
    p.add_argument("--input", action="append", default=[], metavar="STAGE=RUN",
                   help="read STAGE from a run saved under a name (Save as) or from a file, e.g. lifetimes=hot "
                        "or rydberg=/path/Na_rydberg.json; 'STAGE=' goes back to the usual output. Repeatable.")
    p.add_argument("--dpi", type=int, default=600, help="resolution for PNG and TIFF (default 600)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from qali.core.workspace import Workspace
    from qali.figures import recipe as R
    from qali.figures import spec as S
    from qali.figures.templates import TemplateError

    try:
        rec = R.load_recipe(args.recipe)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.species:
        rec["species"] = args.species
    for item in args.input:
        stage, sep, value = item.partition("=")
        if not sep or not stage.strip():
            print(f"--input {item}: give STAGE=RUN, e.g. lifetimes=hot.", file=sys.stderr)
            return 1
        value = value.strip()
        if value and (os.sep in value or "/" in value or value.endswith(".json")):
            value = os.path.abspath(value)
        rec.setdefault("inputs", {})[stage.strip()] = value
    data_dir = args.data_dir or str(Workspace.current().data_json_dir)
    out = args.out or os.path.splitext(args.recipe)[0] + ".pdf"
    try:
        spec = R.resolve(rec, data_dir)
        S.save(spec, out, dpi=args.dpi)
    except (TemplateError, S.RenderError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Figure ({rec['template']}, {rec.get('species') or '-'}) -> {out}")
    for note in spec.notes:
        print(f"  {note}")
    if args.script:
        with open(args.script, "w", encoding="utf-8") as f:
            f.write(S.to_script(spec, os.path.splitext(os.path.basename(out))[0] + ".pdf"))
        print(f"Script -> {args.script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
