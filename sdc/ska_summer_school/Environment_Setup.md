## Environment Setup Instructions

**Create and activate the environment**

```bash
conda create -n ska python=3.8 -y
conda activate ska
```

**Install required packages**

```bash
pip install 21cmfast==3.4.0
pip install tools21cm==2.3.7
pip install natsort==8.4.0
pip install jupyterlab
```

**Clone the Pipeline**

```python
git clone https://github.com/suicee/21cmPipeline.git
```

**Add to Python Path**

At the beginning of your notebook or Python script, add the pipeline to your system path so its modules can be imported:

```python
import sys
sys.path.append("/work/dante/scripts/ska_summer_school/21cmPipeline")
```