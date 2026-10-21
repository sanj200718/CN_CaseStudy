#!/usr/bin/env python3
"""
Mininet topology mirroring the DQN training graph (see graph_config.py).

Run (Ryu must already be running):
    sudo python3 topo.py
Quick test without CLI:
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
                  link=TCLink, autoSetMacs=False)

    net.addController("c0", controller=RemoteController,
                      ip=cfg.CONTROLLER_IP, port=cfg.CONTROLLER_PORT)

    # 6 switches: node i -> s(i+1), explicit dpid so the controller can map it back
    sw = {}
    for n in range(cfg.NUM_NODES):
        sw[n] = net.addSwitch("s%d" % (n + 1),
                              dpid="%016x" % cfg.node_to_dpid(n),
                              protocols="OpenFlow13")

    h1 = net.addHost("h1", ip=cfg.H1_IP + "/24", mac=cfg.H1_MAC)
    h2 = net.addHost("h2", ip=cfg.H2_IP + "/24", mac=cfg.H2_MAC)

    # Host links (small delay, big bandwidth)
    net.addLink(h1, sw[cfg.SRC_NODE], bw=cfg.HOST_LINK_BW, delay="1ms")
    net.addLink(h2, sw[cfg.DST_NODE], bw=cfg.HOST_LINK_BW, delay="1ms")

    # Inter-switch links: one per edge of the training graph
    for (a, b), d in cfg.BASE_DELAYS_MS.items():
        net.addLink(sw[a], sw[b], bw=cfg.LINK_BW_MBPS, delay="%dms" % d)

    return net


def main():
    setLogLevel("info")
    net = build()
    net.start()

    # The graph has loops, so ARP broadcasts would storm. Use static ARP instead;
    # the controller never needs to flood.
    h1, h2 = net.get("h1"), net.get("h2")
    h1.cmd("arp -s %s %s" % (cfg.H2_IP, cfg.H2_MAC))
    h2.cmd("arp -s %s %s" % (cfg.H1_IP, cfg.H1_MAC))

    if "--test" in sys.argv:
        info("*** h1 -> h2 ping\n")
        net.ping([h1, h2])
    else:
        CLI(net)
    net.stop()


if __name__ == "__main__":
    main()
