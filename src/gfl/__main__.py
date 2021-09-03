import argparse
import os
import sys

gfl_abs_path = os.path.dirname(os.path.abspath(__file__))
if os.path.samefile(sys.path[0], gfl_abs_path):
    sys.path.remove(sys.path[0])
gfl_parent_abs_path = os.path.dirname(gfl_abs_path)
if len([sp for sp in sys.path if os.path.exists(sp) and os.path.samefile(gfl_parent_abs_path, sp)]) == 0:
    sys.path.insert(0, gfl_parent_abs_path)

from gfl.application import Application
from gfl.conf import GflConf
from gfl.shell import Shell


parser = argparse.ArgumentParser(prog="GFL")
subparsers = parser.add_subparsers(dest="action", title="actions")

init_parser = subparsers.add_parser("init", help="init gfl env")
init_parser.add_argument("--home", type=str)
init_parser.add_argument("-F", "--force", action="store_true")

app_parser = subparsers.add_parser("run", help="startup gfl")
app_parser.add_argument("--home", type=str)
app_parser.add_argument("--role", type=str)
app_parser.add_argument("--console", action="store_true")
app_parser.add_argument("-D", "--define", dest="props", action="append", type=str)

attach_parser = subparsers.add_parser("attach", help="connect to gfl node")
attach_parser.add_argument("--home", type=str)
attach_parser.add_argument("-H", "--host", type=str, default="127.0.0.1")
attach_parser.add_argument("-P", "--port", type=int, default=9434)


def detect_homedir(args_home):
    if args_home is not None:
        return args_home
    env = os.getenv("GFL_HOME")
    if env is not None:
        return env
    return GflConf.home_dir


if __name__ == "__main__":
    args = parser.parse_args()
    GflConf.home_dir = detect_homedir(args.home)

    if args.action == "attach":
        Shell.attach(args.host, args.port)
    else:
        GflConf.load()
        Shell.welcome()
        if args.action == "init":
            Application.init(args.force)
        elif args.action == "run":
            props = {}
            if args.props:
                for p in args.props:
                    kv = p.split("=")
                    props[kv[0]] = kv[1]
            Application.run(role=args.role, console=args.console, **props)
        else:
            print("unknown action.")
