import dgl
import time
import dgl.function as fn
import torch as th
import pickle
import csv
from contextlib import contextmanager
from dgl.data import load_data, PPIDataset
from collections import namedtuple

class th_op_time(object):
    def __enter__(self):
        self.start_event = th.cuda.Event(enable_timing=True)
        self.end_event = th.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self
        
    def __exit__(self, type, value, traceback):
        self.end_event.record()
        th.cuda.synchronize()  # Wait for the events to be recorded!
        self.time = self.start_event.elapsed_time(self.end_event) / 1000

def get_multi_head_spmm_flops(g, num_hid, num_heads):
    g.ndata['h'] = th.rand(g.number_of_nodes(), num_heads, num_hid).cuda()
    g.edata['w'] = th.rand(g.number_of_edges(), num_heads, 1).cuda()
    accum_time = 0
    accum_FLOPs = 0 
    for t in range(100):
        with th_op_time() as timer:
            g.update_all2(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h1'))
        
        if t >= 30:
            accum_time += timer.time 
            accum_FLOPs += g.number_of_edges() * num_heads * num_hid
    g.ndata.clear()
    g.edata.clear()
    th.cuda.empty_cache()
    return accum_FLOPs / accum_time

def get_spmm_flops(g, num_hid):
    g.ndata['h'] = th.rand(g.number_of_nodes(), num_hid).cuda()
    g.edata['w'] = th.rand(g.number_of_edges(), 1).cuda()
    accum_time = 0
    accum_FLOPs = 0 
    for t in range(100):
        with th_op_time() as timer:
            g.update_all2(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h1'))

        if t >= 30:
            accum_time += timer.time
            accum_FLOPs += g.number_of_edges() * num_hid
    g.ndata.clear()
    g.edata.clear()
    th.cuda.empty_cache()
    return accum_FLOPs / accum_time

def get_multi_head_sddmm_flops(g, num_hid, num_heads):
    g.ndata['h'] = th.rand(g.number_of_nodes(), num_heads, num_hid).cuda()
    g.ndata['x'] = th.rand(g.number_of_nodes(), num_heads, num_hid).cuda()
    accum_time = 0
    accum_FLOPs = 0 
    for t in range(100):
        with th_op_time() as timer:
            g.apply_edges2(fn.u_dot_v('h', 'x', 'h'))

        if t >= 30:
            accum_time += timer.time 
            accum_FLOPs += g.number_of_edges() * num_heads * num_hid
    g.ndata.clear()
    g.edata.clear()
    th.cuda.empty_cache()
    return accum_FLOPs / accum_time

def get_sddmm_flops(g, num_hid):
    g.ndata['h'] = th.rand(g.number_of_nodes(), num_hid).cuda()
    g.ndata['x'] = th.rand(g.number_of_nodes(), num_hid).cuda()
    accum_time = 0
    accum_FLOPs = 0 
    for t in range(100):
        with th_op_time() as timer:
            g.apply_edges2(fn.u_dot_v('h', 'x', 'h'))

        if t >= 30:
            accum_time += timer.time 
            accum_FLOPs += g.number_of_edges() * num_hid
    g.ndata.clear()
    g.edata.clear()
    th.cuda.empty_cache()
    return accum_FLOPs / accum_time

def benchmark(g, dataset):
    f1 = open(dataset + '_spmm.csv', 'w')
    f2 = open(dataset + '_sddmm.csv', 'w')
    print(g, dataset)
    num_heads = 8
    with f1, f2:
        spmm_writer = csv.writer(f1)
        spmm_writer.writerow(['feat size', 'coo 8-head', 'coo', 'csr 8-head', 'csr'])
        sddmm_writer = csv.writer(f2)
        sddmm_writer.writerow(['feat size', 'coo 8-head', 'coo', 'csr 8-head', 'csr'])

        for num_hid in [1, 2, 4, 8, 16, 32, 64, 128]:
            row_spmm = []
            row_sddmm = []
            print('feat size: {}'.format(num_hid))
            row_spmm.append(num_hid)
            row_sddmm.append(num_hid)
            # COO format speed.
            dgl.spfmt.set_format('coo')
            multi_head_spmm = get_multi_head_spmm_flops(g, num_hid, num_heads)
            multi_head_sddmm = get_multi_head_sddmm_flops(g, num_hid, num_heads)
            spmm = get_spmm_flops(g, num_hid)
            sddmm = get_sddmm_flops(g, num_hid)
            print('multi-head spmm FLOPS : ', multi_head_spmm)
            print('multi-head sddmm FLOPS : ', multi_head_sddmm)
            print('spmm FLOPS : ', spmm)
            print('sddmm FLOPS : ', sddmm)
            row_spmm.append(multi_head_spmm / 1e9)
            row_spmm.append(spmm / 1e9)
            row_sddmm.append(multi_head_sddmm / 1e9)
            row_sddmm.append(sddmm / 1e9)
            
            # CSR format speed.
            dgl.spfmt.set_format('csr')
            multi_head_spmm = get_multi_head_spmm_flops(g, num_hid, num_heads)
            multi_head_sddmm = get_multi_head_sddmm_flops(g, num_hid, num_heads)
            spmm = get_spmm_flops(g, num_hid)
            sddmm = get_sddmm_flops(g, num_hid)
            print('multi-head spmm FLOPS : ', multi_head_spmm)
            print('multi-head sddmm FLOPS : ', multi_head_sddmm)
            print('spmm FLOPS : ', spmm)
            print('sddmm FLOPS : ', sddmm)

            row_spmm.append(multi_head_spmm / 1e9)
            row_spmm.append(spmm / 1e9)
            row_sddmm.append(multi_head_sddmm / 1e9)
            row_sddmm.append(sddmm / 1e9)
            spmm_writer.writerow(row_spmm)
            sddmm_writer.writerow(row_sddmm)


if __name__ == '__main__':
    Dataset = namedtuple('Dataset', ['dataset'])
    data = load_data(Dataset(dataset='reddit'))
    benchmark(dgl.graph(data.graph.edges()), 'reddit')
    """
    for K in [8, 16, 32, 64]:
        filename = 'modelnet40_{}.g'.format(K)
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        g = dgl.graph(dgl.batch(data).edges())
        benchmark(g, 'mesh_{}'.format(K))
    """
    """
    ppi = PPIDataset('train')
    benchmark(dgl.graph(ppi.graph.edges()), 'ppi')
    """
    
