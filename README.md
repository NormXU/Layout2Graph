# Paragraph2Graph: A Language-independent GNN-based framework for layout analysis

![Overview of the pipeline](doc/pipeline.png)

This is the official code released repo for [_Paragraph2Graph: A Language-independent GNN-based framework for layout analysis_](https://arxiv.org/pdf/2304.11810.pdf)

## Introduction
we propose a language-independent GNN framework for document layout analysis tasks. Our proposed model, Paragraph2Graph, uses a pre-trained CNN to encode image features and incorporates 2d OCR text coordinates and image features as node features in a graph. We use a dynamic graph convolutional neural network (DGCNN) to update the graph based on these features and include edge features based on relationships. With only 19.95 million parameters, our model is suitable for industrial applications, particularly in multi-language scenarios.

## Usage
Install with `pip install -r requirements.txt`

**Start Training**: `python mytools/train_graph.py`
