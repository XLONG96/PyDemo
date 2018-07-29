'''
Created on 2018年5月17日

@author: Administrator
'''

import pcap
import sys
import string
import time
import socket
import struct

#protocols={socket.IPPROTO_TCP:'tcp',
           #socket.IPPROTO_UDP:'udp',
           #socket.IPPROTO_ICMP:'icmp'}
protocols={
    0x00:"HOPOPT",
    0x01:"ICMP",
    0x02:"IGMP",
    0x03:"GGP",
    0x04:"IP-in-IP",
    0x05:"ST",
    0x06:"TCP",
    0x07:"CBT",
    0x08:"EGP",
    0x09:"IGP",
    0x0A:"BBN-RCC-MON",
    0x0B:"NVP-II",
    0x0C:"PUP",
    0x0D:"ARGUS",
    0x0E:"EMCON",
    0x0F:"XNET",
    0x10:"CHAOS",
    0x11:"UDP",
    0x12:"MUX",
    0x13:"DCN-MEAS",
    0x14:"HMP",
    0x15:"PRM",
    0x16:"XNS-IDP",
    0x17:"TRUNK-1",
    0x18:"TRUNK-2",
    0x19:"LEAF-1",
    0x1A:"LEAF-2",
    0x1B:"RDP",
    0x1C:"IRTP",
    0x1D:"ISO-TP4",
    0x1E:"NETBLT",
    0x1F:"MFE-NSP",
    0x20:"MERIT-INP",
    0x21:"DCCP",
    0x22:"3PC",
    0x23:"IDPR",
    0x24:"XTP",
    0x25:"DDP",
    0x26:"IDPR-CMTP",
    0x27:"TP++",
    0x28:"IL",
    0x29:"IPv6",
    0x2A:"SDRP",
    0x2B:"IPv6-Route",
    0x2C:"IPv6-Frag",
    0x2D:"IDRP",
    0x2E:"RSVP",
    0x2F:"GRE",
    0x30:"MHRP",
    0x31:"BNA",
    0x32:"ESP",
    0x33:"AH",
    0x34:"I-NLSP",
    0x35:"SWIPE",
    0x36:"NARP",
    0x37:"MOBILE",
    0x38:"TLSP",
    0x39:"SKIP",
    0x3A:"IPv6-ICMP",
    0x3B:"IPv6-NoNxt",
    0x3C:"IPv6-Opts",
    0x3D:"host internal protocol", #any
    0x3E:"CFTP",
    0x3F:"local network", #any 
    0x40:"SAT-EXPAK",
    0x41:"KRYPTOLAN",
    0x42:"RVD",
    0x43:"IPPC",
    0x44:"distributed file system", #any 
    0x45:"SAT-MON", 
    0x46:"VISA", 
    0x47:"IPCU", 
    0x48:"CPNX", 
    0x49:"CPHB", 
    0x4A:"WSN", 
    0x4B:"PVP", 
    0x4C:"BR-SAT-MON", 
    0x4D:"SUN-ND", 
    0x4E:"WB-MON", 
    0x4F:"WB-EXPAK", 
    0x50:"ISO-IP", 
    0x51:"VMTP", 
    0x52:"SECURE-VMTP", 
    0x53:"VINES", 
    0x54:"TTP", 
    0x54:"IPTM", 
    0x55:"NSFNET-IGP", 
    0x56:"DGP", 
    0x57:"TCF", 
    0x58:"EIGRP", 
    0x59:"OSPF", 
    0x5A:"Sprite-RPC", 
    0x5B:"LARP", 
    0x5C:"MTP", 
    0x5D:"AX.25", 
    0x5E:"IPIP", 
    0x5F:"MICP", 
    0x60:"SCC-SP", 
    0x61:"ETHERIP", 
    0x62:"ENCAP", 
    0x63:"", 
    0x64:"GMTP", 
    0x65:"IFMP", 
    0x66:"PNNI", 
    0x67:"PIM", 
    0x68:"ARIS", 
    0x69:"SCPS", 
    0x6A:"QNX", 
    0x6B:"A/N", 
    0x6C:"IPComp", 
    0x6D:"SNP", 
    0x6E:"Compaq-Peer", 
    0x6F:"IPX-in-IP", 
    0x70:"VRRP", 
    0x71:"PGM", 
    0x72:"", 
    0x73:"L2TP", 
    0x74:"DDX", 
    0x75:"IATP", 
    0x76:"STP", 
    0x77:"SRP", 
    0x78:"UTI", 
    0x79:"SMP", 
    0x7A:"SM", 
    0x7B:"PTP", 
    0x7C:"IS-IS over IPv4", 
    0x7D:"FIRE", 
    0x7E:"CRTP", 
    0x7F:"CRUDP", 
    0x80:"SSCOPMCE", 
    0x81:"IPLT", 
    0x82:"SPS", 
    0x83:"PIPE", 
    0x84:"SCTP", 
    0x85:"FC", 
    0x86:"RSVP-E2E-IGNORE", 
    0x87:"Mobility Header", 
    0x88:"UDPLite", 
    0x89:"MPLS-in-IP", 
    0x8A:"manet", 
    0x8B:"HIP", 
    0x8C:"Shim6", 
    0x8D:"WESP", 
    0x8E:"ROHC", 
}

import socket
socket.inet_ntoa
def decode_ip_packet(s):
    d={}
    d['version']=(ord(s[0]) & 0xf0) >> 4
    d['header_len']=ord(s[0]) & 0x0f
    d['tos']=ord(s[1])
    d['total_len']=socket.ntohs(struct.unpack('H',s[2:4])[0])
    d['id']=socket.ntohs(struct.unpack('H',s[4:6])[0])
    d['flags']=(ord(s[6]) & 0xe0) >> 5
    d['fragment_offset']=socket.ntohs(struct.unpack('H',s[6:8])[0] & 0x1f)
    d['ttl']=ord(s[8])
    d['protocol']=ord(s[9])
    d['checksum']=socket.ntohs(struct.unpack('H',s[10:12])[0])
    d['source_address']=socket.inet_ntoa(s[12:16]) 
    d['destination_address']=socket.inet_ntoa(s[16:20])
    if d['header_len']>5:
        d['options']=s[20:4*(d['header_len']-5)]
    else:
        d['options']=None
    d['data']=s[4*d['header_len']:]
    return d


def dumphex(s):
    bytes = map(lambda x: '%.2x' % x, map(ord, s))
    for i in xrange(0,len(bytes)/16):
        print ('    %s' % string.join(bytes[i*16:(i+1)*16],' '))
        print ('    %s' % string.join(bytes[(i+1)*16:],' '))


def print_packet( data, timestamp):
    if not data:
        return
    if data[12:14]=='\x08\x00': #IP 包
        decoded=decode_ip_packet(data[14:])
        print ('\n%s.%f %s > %s' % (time.strftime('%H:%M',
                                                 time.localtime(timestamp)),
                                   timestamp % 60,
                                   decoded['source_address'],
                                   decoded['destination_address']))
        for key in ['version', 'header_len', 'tos', 'total_len', 'id',
                    'flags', 'fragment_offset', 'ttl']:
            print ('  %s: %d' % (key, decoded[key]))
        print ('  protocol: %s' % protocols[decoded['protocol']])
        print ('  header checksum: %d' % decoded['checksum'])
        #print '  data:'
        #dumphex(decoded['data'])


if __name__=='__main__':
    print ( pcap.findalldevs() )
    for dev in pcap.findalldevs():
        print (dev)
        net, mask = pcap.lookupnet(dev.encode())
        print (net.__repr__(),mask.__repr__())
    p = pcap.pcap('本地网络')
    net, mask = pcap.lookupnet(dev.encode())
    try:
        for timestamp, data in p:
            print_packet(data, timestamp)
            #print timestamp, len(data)
    except KeyboardInterrupt:
        print ('%s' % sys.exc_type)
        print ('shutting down')
        
        
