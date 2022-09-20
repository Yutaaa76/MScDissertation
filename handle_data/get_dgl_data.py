from mumin import MuminDataset
import dgl
import torch


def get_data(size):
    dataset = MuminDataset(size=size,
                           twitter_bearer_token='AAAAAAAAAAAAAAAAAAAAAD18aQEAAAAA9cvR2Es1C7oE%2BYSdikWfevjKk6Y%3DCEpsydB03s8C6JXvZdiZEGWLMlHk8Py4lfBEFFYNieH6L5yXuq')
    dataset.compile()
    dgl.seed(990706)
    if 'dgl_graph' not in globals():
        dgl_graph = dataset.to_dgl()

    # remove 'reply' and 'hashtag' nodes
    dgl_graph = dgl_graph.node_type_subgraph(['tweet', 'user', 'claim'])

    # remove 'reply' nodes (too many to calculate)
    # dgl_graph = dgl.remove_nodes(dgl_graph, dgl_graph.nodes('reply'), ntype='reply')
    # for edge in dgl_graph.canonical_etypes:
    #     if 'reply' in edge:
    #         dgl_graph = dgl.remove_edges(dgl_graph,
    #                                      torch.Tensor([i for i in range(len(dgl_graph.edges(etype=edge)[0]))]).int(), edge)
    return dgl_graph


# data = get_data('small')
# print(data)
