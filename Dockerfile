FROM nsml/default_ml:cuda9_torch1.0

RUN pip install torch==1.1.0 torchvision==0.3.0 pretrainedmodels tqdm sklearn pyyaml
