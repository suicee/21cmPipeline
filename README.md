# 21cmPipeline

A simple 21cm pipeline developed to generate mock 21cm observations for inference tasks and other machine learning applications in my research. The pipeline covers several fundamental processes:  
- Generating 21cm brightness temperature simulations and constructing lightcones  
- Foreground simulation and mitigation  
- Calculation of summary statistics  

Most functions are simple wrappers around well-established packages such as [tools21cm](https://github.com/sambit-giri/tools21cm), [21cmFAST](https://github.com/21cmfast/21cmFAST), and [pygdsm](https://github.com/telegraphic/pygdsm).

This pipeline was used to produce results for the SDC challenges with teams [Shuimu-Tianlai-A, Shuimu-Tianlai-B, and Shuimu-Tianlai-C](https://sdc3.skao.int/challenges/inference/results), where we achieved reasonable inference outcomes. The full SDC workflow is documented in detail across three notebooks located in the `sdc` directory.
