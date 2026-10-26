#!/usr/bin/env python3
"""Mininet topology matching graph_config.py and Sanjana's six-node graph.

Start the Ryu controller first, then run:
    sudo python3 topo.py
Quick connectivity smoke test (requires controller):
    sudo python3 topo.py --test
"""
import sys
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info

import graph_config as cfg


def build():
    net = Mininet(controller=None, switch=OVSKernelSwitch,
                  link=TCLink, autoSetMacs=False, build=False)

    net.addController("c0", controller=RemoteController,
                      ip=cfg.CONTROLLER_IP, port=cfg.CONTROLLER_PORT)

    switches = {}
    for node in range(cfg.NUM_NODES):
        switches[node] = net.addSwitch(
            cfg.switch_name(node),
            dpid=f"{cfg.node_to_dpid(node):016x}",
            protocols="OpenFlow13",
        )

    h1 = net.addHost("h1", ip=cfg.H1_IP + "/24", mac=cfg.H1_MAC)
    h2 = net.addHost("h2", ip=cfg.H2_IP + "/24", mac=cfg.H2_MAC)

    # Explicit switch-facing host ports prevent controller/topology mismatches.
    net.addLink(h1, switches[cfg.SRC_NODE], port2=cfg.HOST_PORT,
                bw=cfg.HOST_LINK_BW, delay="1ms")
    net.addLink(h2, switches[cfg.DST_NODE], port2=cfg.HOST_PORT,
                bw=cfg.HOST_LINK_BW, delay="1ms")

    inter_switch_links = {}
    for (u, v), delay_ms in cfg.BASE_DELAYS_MS.items():
        pu, pv = cfg.ports_for_edge(u, v)
        link = net.addLink(
            switches[u], switches[v], port1=pu, port2=pv,
            bw=cfg.LINK_BW_MBPS, delay=f"{delay_ms}ms",
        )
        inter_switch_links[cfg.normalized_edge(u, v)] = link

    # Used by experiment scripts after net.start().
    net._routing_switches = switches
    net._routing_inter_switch_links = inter_switch_links
    net._routing_hosts = {"h1": h1, "h2": h2}
    return net


def configure_static_arp(net):
    """The two-host topology uses static ARP to avoid L2 broadcast storms."""
    h1, h2 = net.get("h1", "h2")
    h1_dev = h1.defaultIntf().name
    h2_dev = h2.defaultIntf().name
    h1.cmd(f"ip neigh replace {cfg.H2_IP} lladdr {cfg.H2_MAC} dev {h1_dev}")
    h2.cmd(f"ip neigh replace {cfg.H1_IP} lladdr {cfg.H1_MAC} dev {h2_dev}")


def main():
    setLogLevel("info")
    net = build()
    try:
        net.start()
        configure_static_arp(net)
        if "--test" in sys.argv:
            info("*** h1 -> h2 ping connectivity test\n")
            loss = net.ping([net.get("h1"), net.get("h2")], timeout="2")
            info(f"*** ping packet-loss result: {loss:.1f}%\n")
            if loss >= 100:
                raise SystemExit("Connectivity test failed; check controller logs and OpenFlow rules.")
        else:
            CLI(net)
    finally:
        net.stop()


if __name__ == "__main__":
    main()
