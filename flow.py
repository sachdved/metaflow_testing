import metaflow
import torch
import numpy as np
import scipy as sp
import sklearn
import pandas as pd
import toml
import logging

model_type = {
    'Linear': torch.nn.Linear,
    'ReLU': torch.nn.ReLU,
    'SoftMax': torch.nn.Softmax
}

optimizer = {
    'SGD': torch.optim.SGD
}

class MNIST_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_data
    ):
        self.data = input_data
        
    def __len__(
        self
    ):
        return self.data.shape[0]

    def __getitem__(
        self,
        index
    ):
        label = torch.zeros(10)
        label[self.data.iloc[index, 0]] = 1.
        image = self.data.iloc[index, 1:]
        return {
            'image' : torch.Tensor(image),
            'label' : label
        }

class LinearFlow(metaflow.FlowSpec):
    """
    This is the base class for building a Flow. These are intended to be standardized.
    This ensures reproducibility, tracking, and should be relatively easy to use.
    """

    config_path = metaflow.Parameter(
        'config',
        help = "TOML file with all configurable parameters",
        default = None,
        required = True
    )
    
    @metaflow.step
    def start(self):
        """
        This is the start of the metaflow run. Start by
        loading in the data and storing various attributes.
        """

        self.config = toml.load(self.config_path)
        self.train_data = pd.read_csv(self.config['data']['train_path'])
        self.test_data = pd.read_csv(self.config['data']['test_path'])
        self.next(self.dataloaders)

    @metaflow.step
    def dataloaders(
        self
    ):
        """
        Instantiate the dataloaders.
        """
        self.batch_size = self.config['dataloaders']['batch_size']
        self.train_dataset = MNIST_dataset(
            self.train_data
        )
        self.test_dataset = MNIST_dataset(
            self.test_data
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False
        )
        
        for batch in self.train_dataloader:
            batch
        for batch in self.test_dataloader:
            batch
        
        self.next(self.train)

    @metaflow.step
    def train(
        self
    ):
        """
        Construct and train the model.
        """
        layers = []
        for key_stuff in self.config['model']['layers'].keys():
            layer_type = model_type[self.config['model']['layers'][key_stuff]['type']]
            layer_args = {
                key : value for (key, value) in self.config['model']['layers'][key_stuff].items() if key != 'type'
            }
            layers.append(
                layer_type(**layer_args)
            )
        self.model = torch.nn.Sequential(*layers)
        self.optim = optimizer[self.config['model']['optimizer']['type']]
        optim_args = {
                key : value for (key, value) in self.config['model']['optimizer'].items() if key != 'type'
            }
        optim_args['params'] = self.model.parameters()
        self.optim = self.optim(**optim_args)

        for epoch in range(100):
            
            for batch in self.train_dataloader:
                
                self.optim.zero_grad()
                pred = self.model(batch['image'])
                loss = torch.mean((pred - batch['label'])**2)
                loss.backward()
                self.optim.step()
        self.next(self.end)

    @metaflow.step
    def end(
        self
    ):
        logging.info("We're done!")
    

if __name__ == "__main__":
    LinearFlow()