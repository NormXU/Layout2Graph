# Paragraph2Graph: A Language-independent GNN-based framework for layout analysis (WIP 🚧)

![Overview of the pipeline](doc/pipeline.png)

This is the official code released repo for [_Paragraph2Graph: A Language-independent GNN-based framework for layout analysis_](https://arxiv.org/pdf/2304.11810.pdf)

## Introduction
we propose a language-independent GNN framework for document layout analysis tasks. Our proposed model, Paragraph2Graph, uses a pre-trained CNN to encode image features and incorporates 2d OCR text coordinates and image features as node features in a graph. We use a dynamic graph convolutional neural network (DGCNN) to update the graph based on these features and include edge features based on relationships. With only 19.95 million parameters, our model is suitable for industrial applications, particularly in multi-language scenarios.

## How to Use
### 1. Set up the environment
If you are using pytorch==2.0.1, you can simply set up the environment by 
```json
pip install -r requirements.txt
```

If you are using an older pytorch==1.12.1, make sure to edit ```requirements.txt``` by replacing 
```-r requirements/requirements-dependency-2.0.1.txt``` with ```-r requirements/requirements-dependency-1.12.1.txt```

### 2. Prepare dataset

If you find our work helpful, please consider citing our work and leaving us a star.
```
@article{wei2023paragraph2graph,
  title={PARAGRAPH2GRAPH: A GNN-based framework for layout paragraph analysis},
  author={Wei, Shu and Xu, Nuo},
  journal={arXiv preprint arXiv:2304.11810},
  year={2023}
}
```
