# Layout2Graph: A Language-independent GNN-based framework for layout analysis

![Overview of the pipeline](doc/pipeline.png)

This is the official code released repo for [_Layout2Graph: A Language-independent GNN-based framework for layoutanalysis_](doc/Layout2Graph__A_Language_independent_GNN_model__preprint_.pdf)

## Introduction
we propose a language-independent GNN framework for document layout analysis tasks. Our proposed model, Layout2Graph, uses a pre-trained CNN to encode image features and incorporates 2d OCR text coordinates and image features as node features in a graph. We use a dynamic graph convolutional neural network (DGCNN) to update the graph based on these features and include edge features based on relationships.

## Usage
Install with `pip install -r requirements.txt`

**Start Training**: `python mytools/train_graph.py`
