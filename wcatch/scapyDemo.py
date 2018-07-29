'''
Created on 2018年5月18日

@author: Administrator
'''
import os  
from scapy.all import *  
import dpkt

pkts=[]  
count=0  
pcapnum=0  
filename=''  
  
def test_dump_file(dump_file):  
    print ("Testing the dump file..." ) 
     
    if os.path.exists(dump_file):  
        print ("dump fie %s found." %dump_file)  
        pkts=sniff(offline=dump_file)  
        count = 0  
        while (count<=2):                                       
            print ("----Dumping pkt:%s----" %dump_file ) 
            print (hexdump(pkts[count]))  
            count +=1  
    else:  
        print ("dump fie %s not found." %dump_file ) 
  
def write_cap(x):  
    global pkts  
    global count  
    global pcapnum  
    global filename  
    pkts.append(x)  
    count +=1  
    if count ==3:
        pcapnum +=1  
        pname="pcap%d.pcap"%pcapnum  
        wrpcap(pname,pkts)  
        filename ="./pcap%d.pcap"%pcapnum  
        test_dump_file(filename)  
        pkts=[]  
        count=0  
          
  
if __name__=='__main__':  
    print ("Start packet capturing and dumping ..."  )
    #a=sniff(filter="tcp",
    #prn=lambda x: x.sprintf("%IP.src%:%TCP.sport% -> %IP.dst%:%TCP.dport%  %2s,TCP.flags% : %TCP.payload%")) 
    a = sniff(filter = "tcp")
    for timestamp, buf in a:

        # Print out the timestamp in UTC
        print ('Timestamp: ', str(datetime.datetime.utcfromtimestamp(timestamp)))
    
        # Unpack the Ethernet frame (mac src/dst, ethertype)
        eth = dpkt.ethernet.Ethernet(buf)
        print ('Ethernet Frame: ', mac_addr(eth.src), mac_addr(eth.dst), eth.type)
    
        # Make sure the Ethernet frame contains an IP packet
        if not isinstance(eth.data, dpkt.ip.IP):
            print ('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
            continue
    
        # Now unpack the data within the Ethernet frame (the IP packet)
        # Pulling out src, dst, length, fragment info, TTL, and Protocol
        ip = eth.data
    
        # Pull out fragment information (flags and offset all packed into off field, so use bitmasks)
        do_not_fragment = bool(ip.off & dpkt.ip.IP_DF)
        more_fragments = bool(ip.off & dpkt.ip.IP_MF)
        fragment_offset = ip.off & dpkt.ip.IP_OFFMASK
    
        # Print out the info
        print ('IP: %s -> %s   (len=%d ttl=%d DF=%d MF=%d offset=%d)\n' % \
              (inet_to_str(ip.src), inet_to_str(ip.dst), ip.len, ip.ttl, do_not_fragment, more_fragments, fragment_offset))
           