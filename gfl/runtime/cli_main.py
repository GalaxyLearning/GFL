import argparse

from .action import gfl_init, gfl_start, gfl_attach


def parse_args(args=None):
    parser = argparse.ArgumentParser(prog="GFL", description="")
    subparsers = parser.add_subparsers(dest="action", title="action")

    init_parser = subparsers.add_parser("init", help="init gfl_p env")
    init_parser.add_argument("--home", type=str, required=False, help="gfl_p home directory")
    init_parser.add_argument("--gfl_p-config", "--gfl_config", type=str, required=False, help="gfl_p config file path")
    init_parser.add_argument("--mode", type=str, required=False, help="node communication mode")
    init_parser.add_argument("--as-http-server", action="store_true", help="make this node as http server")
    init_parser.add_argument("--http-ip", type=str, default="127.0.0.1", required=False, help="http server ip")
    init_parser.add_argument("--http-port", type=int, default=10700, required=True, help="http server port")
    init_parser.add_argument("--init-eth", action="store_true", help="deploy contracts")
    init_parser.add_argument("--eth-ip", type=str, default="127.0.0.1", required=False, help="eth node ip")
    init_parser.add_argument("--eth-port", type=int, default=8545, required=False, help="eth node port")
    init_parser.add_argument("--force", action="store_true", help="overwrite exists directory")

    start_parser = subparsers.add_parser("start", help="start gfl_p node")
    start_parser.add_argument("--home", type=str, required=False, help="gfl_p home directory")
    start_parser.add_argument("--allow-remote", action="store_true", help="whether allow remote connection")
    start_parser.add_argument("--no-webui", action="store_true", help="disable webui")
    start_parser.add_argument("--no-daemon", action="store_true", help="start without daemon")
    start_parser.add_argument("--shell", action="store_true", help="start shell with main process")

    attach_parser = subparsers.add_parser("attach", help="attach to gfl_p node")
    attach_parser.add_argument("--shell-type", type=str, default="ipython", help="shell type")
    attach_parser.add_argument("--home", type=str, required=False, help="gfl_p home directory")
    attach_parser.add_argument("--node-ip", type=str, required=False, help="gfl_p node ip")
    attach_parser.add_argument("--node-port", type=str, required=False, help="gfl_p node port")

    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    if args.action == "init":
        gfl_init(home=args.home,
                 gfl_config=args.gfl_config,
                 force=args.force)
    elif args.action == "start":
        gfl_start(home=args.home,
                  no_webui=args.no_webui,
                  no_daemon=args.no_daemon,
                  shell=args.shell)
    elif args.action == "attach":
        gfl_attach(shell_type=args.shell_type,
                   home=args.home,
                   http_ip=args.node_ip,
                   http_port=args.node_ip)
        pass
    else:
        raise ValueError(f"Unsupported action {args.action}")
