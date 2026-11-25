import argparse, sys
from . import PunctRestorer, __version__


def main(argv=None):
    p = argparse.ArgumentParser(prog="punct-restore",
        description="Restore punctuation and capitalization. Reads TEXT, or one text per line from stdin.")
    p.add_argument("text", nargs="*", help="text to restore (omit to read lines from stdin)")
    p.add_argument("-m", "--model", default="small", help='"small" (default), "base", a HF model id or a local dir')
    p.add_argument("--device", default=None)
    p.add_argument("--version", action="version", version=__version__)
    a = p.parse_args(argv)
    pr = PunctRestorer(a.model, device=a.device)
    if a.text:
        print(pr(" ".join(a.text)))
    else:
        for line in sys.stdin:
            print(pr(line.rstrip("\n")))


if __name__ == "__main__":
    main()
