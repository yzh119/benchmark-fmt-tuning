import dgl
import time
import dgl.function as fn
import torch as th
import pickle
import csv
from dgl.data import load_data, PPIDataset
from collections import namedtuple

def get_multi_head_spmm_flops(g, num_hid, num_heads):
    g.ndata['h'] = th.rand(g.number_of_nodes(), num_heads, num_hid).cuda()
    g.edata['w'] = th.rand(g.number_of_edges(), num_heads, 1).cuda()
    accum_time = 0
    accum_FLOPs = 0 
    for t in range(100):
        th.cuda.synchronize()
        tic = time.time()
        with th.no_grad():
            g.update_all2(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h1'))
        th.cuda.synchronize()
        toc = time.time()
        
        if t >= 30:
            accum_time += toc - tic
            accum_FLOPs += g.number_of_edges() * num_heads * num_hid
    g.ndata.clear()
    g.edata.clear()
    return accum_FLOPs / accum_time

def get_spmm_flops(g, num_hid):
    g.ndata['h'] = th.rand(g.number_of_nodes(), num_hid).cuda()
    g.edata['w'] = th.rand(g.number_of_edges(), 1).cuda()
    accum_time = 0
    accum_FLOPs = 0 
    for t in range(100):
        th.cuda.synchronize()
        tic = time.time()
        with th.no_grad():
            g.update_all2(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h1'))
        th.cuda.synchronize()
        toc = time.time()
        
        if t >= 30:
            accum_time += toc - tic
            accum_FLOPs += g.number_of_edges() * num_hid
    g.ndata.clear()
    g.edata.clear()
    return accum_FLOPs / accum_time

def get_multi_head_sddmm_flops(g, num_hid, num_heads):
    g.ndata['h'] = th.rand(g.number_of_nodes(), num_heads, num_hid).cuda()
    g.ndata['x'] = th.rand(g.number_of_nodes(), num_heads, num_hid).cuda()
    accum_time = 0
    accum_FLOPs = 0 
    for t in range(100):
        th.cuda.synchronize()
        tic = time.time()
        with th.no_grad():
            g.apply_edges2(fn.u_dot_v('h', 'x', 'h'))
        th.cuda.synchronize()
        toc = time.time()

        if t >= 10:
            accum_time += toc - tic
            accum_FLOPs += g.number_of_edges() * num_heads * num_hid
    g.ndata.clear()
    g.edata.clear()
    return accum_FLOPs / accum_time

def get_sddmm_flops(g, num_hid):
    g.ndata['h'] = th.rand(g.number_of_nodes(), num_hid).cuda()
    g.ndata['x'] = th.rand(g.number_of_nodes(), num_hid).cuda()
    accum_time = 0
    accum_FLOPs = 0 
    for t in range(100):
        th.cuda.synchronize()
        tic = time.time()
        with th.no_grad():
            g.apply_edges2(fn.u_dot_v('h', 'x', 'h'))
        th.cuda.synchronize()
        toc = time.time()

        if t >= 10:
            accum_time += toc - tic
            accum_FLOPs += g.number_of_edges() * num_hid
    g.ndata.clear()
    g.edata.clear()
    return accum_FLOPs / accum_time

def benchmark(g, dataset):
    f = open(dataset + '.csv', 'w')
    print(g)
    num_heads = 8
    with f:
        writer = csv.writer(f)
        writer.writerow(['feat size', 'coo 8-head', 'coo', 'csr 8-head', 'csr'])

        for num_hid in [1, 2, 4, 8, 16, 32, 64, 128]:
            row = []
            print('feat size: {}'.format(num_hid))
            row.append(num_hid)
            # COO format speed.
            dgl.spfmt.set_format('coo')
            multi_head_spmm = get_multi_head_spmm_flops(g, num_hid, num_heads)
            #multi_head_sddmm_time = get_multi_head_sddmm_flops(g, num_hid, num_heads)
            spmm = get_spmm_flops(g, num_hid)
            #sddmm_time = get_sddmm_flops(g, num_hid)
            print('multi-head spmm FLOPS : ', multi_head_spmm)
            #print('multi-head sddmm FLOPS : ', multi_head_sddmm_time)
            print('spmm FLOPS : ', spmm)
            #print('sddmm FLOPS : ', sddmm_time)
            row.append(multi_head_spmm / 1e9)
            row.append(spmm / 1e9)
            
            # CSR format speed.
            dgl.spfmt.set_format('csr')
            multi_head_spmm = get_multi_head_spmm_flops(g, num_hid, num_heads)
            #multi_head_sddmm_time = get_multi_head_sddmm_flops(g, num_hid, num_heads)
            spmm = get_spmm_flops(g, num_hid)
            #sddmm_time = get_sddmm_flops(g, num_hid)
            print('multi-head spmm FLOPS : ', multi_head_spmm)
            #print('multi-head sddmm FLOPS : ', multi_head_sddmm_time)
            print('spmm FLOPS : ', spmm)
            #print('sddmm FLOPS : ', sddmm_time)

            row.append(multi_head_spmm / 1e9)
            row.append(spmm / 1e9)
            writer.writerow(row)


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

    
