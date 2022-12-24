# encoding: utf-8
'''
@email: weishu@datagrand.com
@software: Pycharm
@time: 2021/11/24 6:27 下午
@desc:
'''
import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d

from torch_geometric.data import Batch as GraphBatch
from torch_geometric.data import Data as GraphData


def build_graph(cell_boxes, fusion_feat, device):
    num_images = len(cell_boxes)
    graphs = []
    start_index = 0
    for img_id in range(num_images):
        num_nodes = cell_boxes[img_id].shape[0]
        tb_graph = GraphData(x=fusion_feat[start_index:start_index + num_nodes], pos=cell_boxes[img_id])
        graphs.append(tb_graph)
        start_index += num_nodes
    graphs = GraphBatch.from_data_list(graphs).to(device)
    return graphs

def MLP(channels, batch_norm=True):
    return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), ReLU(), BatchNorm1d(channels[i])) for i in range(1, len(channels))])


class DGCNNModule(torch.nn.Module):

    def __init__(self, input_channels, out_channels, num_neighbors=20, aggr='max', **kwargs):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * input_channels, 64, 64, 64]), num_neighbors, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), num_neighbors, aggr)
        # self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), num_neighbors, aggr)
        # self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), num_neighbors, aggr)
        self.lin1 = MLP([128 + 64, out_channels])


    def forward(self, cell_boxes, fusion_feat):
        self.device = fusion_feat.device
        graphs = build_graph(cell_boxes, fusion_feat, self.device)
        # x0 = torch.cat([graphs.x, graphs.pos], dim=-1)
        x0 = graphs.x
        x1 = self.conv1(x0, graphs.batch)#box_len*64
        x2 = self.conv2(x1, graphs.batch)#box_len*128
        out = self.lin1(torch.cat([x1, x2], dim=1)) #box_len*512
        return out, graphs
