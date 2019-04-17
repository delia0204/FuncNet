# FuncNet
binary association analysis based on the knowledge base, which recognize those duplicated or known parts of binary compiled from various platforms.

## dataset:
    1.extend dataset from Gemimi
    2.dataset includes three kinds of semantic types (i.e., loop, branch, and interaction)
    3.dateset from four real-world projets (i.e., zlib, lua, gzip, curl)
    
## prototype:
### Features:
    basic block attributes based on IDA interface
    control flow graph based on IDA interface
    callee's interface based on function frame and registers' use-before-write feature

### Graph embedding neural network based on structure2vec
trained model in saved_model, just test on it.

### Popsom based on Self-orgnization model
trained model in saved_model, test on it.
