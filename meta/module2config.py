import torch.nn as nn
import torch 
import torch.nn.functional as F

class Config2Module():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    
    def conv_module(self, config):

        assert config['name'] == torch.nn.Conv2d().__class__.__name__, \
            'Expected Conv2d, got {}'.format(config['name'])
        return torch.nn.Conv2d(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            kernel_size=config['kernel_size'],
            stride=config['stride'],
            padding=config['padding'],
            dilation=config['dilation'],
            groups=config['groups'],
            bias=config['bias'],
            padding_mode=config['padding_mode'],
        )
    
    def linear_module(self, config):
        
        assert config['name'] == torch.nn.Linear().__class__.__name__, \
            'Expected Linear, got {}'.format(config['name'])
        return torch.nn.Linear(
            in_features=config['in_features'],
            out_features=config['out_features'],
            bias=config['bias'],
        )
    
    def batchnorm_module(self, config):
            
        assert config['name'] == torch.nn.BatchNorm2d().__class__.__name__, \
            'Expected BatchNorm2d, got {}'.format(config['name'])

        return torch.nn.BatchNorm2d(
            num_features=config['num_features'],
            eps=config['eps'],
            momentum=config['momentum'],
            affine=config['affine'],
            track_running_stats=config['track_running_stats'],
        )

    def activation_module(self, config):
        assert config['name'] == torch.nn.ReLU().__class__.__name__, \
            'Expected ReLU, got {}'.format(config['name'])
        return torch.nn.ReLU()

    def maxpool_module(self, config):

        assert config['name'] == torch.nn.MaxPool2d().__class__.__name__, \
            'Expected MaxPool2d, got {}'.format(config['name'])
        return torch.nn.MaxPool2d(
            kernel_size=config['kernel_size'],
            stride=config['stride'],
            padding=config['padding'],
            dilation=config['dilation'],
            return_indices=config['return_indices'],
            ceil_mode=config['ceil_mode'],
        )

    def avgpool_module(self, config):
            
            assert config['name'] == torch.nn.AvgPool2d().__class__.__name__, \
                'Expected AvgPool2d, got {}'.format(config['name'])
            return torch.nn.AvgPool2d(
                kernel_size=config['kernel_size'],
                stride=config['stride'],
                padding=config['padding'],
                ceil_mode=config['ceil_mode'],
                count_include_pad=config['count_include_pad'],
            )
    
    def dropout_module(self, config):
        assert config['name'] == torch.nn.Dropout().__class__.__name__, \
            'Expected Dropout, got {}'.format(config['name'])
        return torch.nn.Dropout(
            p=config['p'],
            inplace=config['inplace'],
        )
    
    def sequential_module(self, config):
        assert config['name'] == torch.nn.Sequential().__class__.__name__, \
            'Expected Sequential, got {}'.format(config['name'])
        modules = []
        for module_config in config['modules']:
            modules.append(self.convert(module_config))
        return torch.nn.Sequential(*modules)
    
    def module_from_config(self, config):

        if config['name'] == torch.nn.Conv2d().__class__.__name__:
            return self.conv_module(config)
        elif config['name'] == torch.nn.Linear().__class__.__name__:
            return self.linear_module(config)
        elif config['name'] == torch.nn.BatchNorm2d().__class__.__name__:
            return self.batchnorm_module(config)
        elif config['name'] == torch.nn.ReLU().__class__.__name__:
            return self.activation_module(config)
        elif config['name'] == torch.nn.MaxPool2d().__class__.__name__:
            return self.maxpool_module(config)
        elif config['name'] == torch.nn.AvgPool2d().__class__.__name__:
            return self.avgpool_module(config)
        elif config['name'] == torch.nn.Dropout().__class__.__name__:
            return self.dropout_module(config)
        elif config['name'] == torch.nn.Sequential().__class__.__name__:
            return self.sequential_module(config)
        else:
            raise ValueError('Unknown module type: {}'.format(config['name']))
        
    def convert(self, config):
        return self.module_from_config(config)

class Module2Config():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def conv_config(self, module):
        #in_channels, out_channels, kernel_size, stride, padding,
        #dilation, groups, bias, padding_mode
        
        assert isinstance(module, torch.nn.Conv2d), 'Not a Convolutional layer'
        return {
            'name': module.__class__.__name__,
            'in_channels': module.in_channels,
            'out_channels': module.out_channels,
            'kernel_size': module.kernel_size,
            'stride': module.stride,
            'padding': module.padding,
            'dilation': module.dilation,
            'groups': module.groups,
            'bias': module.bias is not None,
            'padding_mode': module.padding_mode,
        }

    def linear_config(self, module):
        #in_features, out_features, bias
        assert isinstance(module, torch.nn.Linear), 'Not a Linear layer'
        return {
            'name': module.__class__.__name__,
            'in_features': module.in_features,
            'out_features': module.out_features,
            'bias': module.bias is not None,
        }
    
    def batchnorm_config(self, module):

        assert isinstance(module, torch.nn.BatchNorm2d), 'Not a BatchNorm layer'
        return {
            'name': module.__class__.__name__,
            'num_features': module.num_features,
            'eps': module.eps,
            'momentum': module.momentum,
            'affine': module.affine,
            'track_running_stats': module.track_running_stats,
        }
    
    def activation_config(self, module):
        assert isinstance(module, torch.nn.ReLU), 'Not a ReLU layer'
        return {
            'name': module.__class__.__name__,
        }
    
    def maxpool_config(self, module):
        assert isinstance(module, torch.nn.MaxPool2d), 'Not a MaxPool layer'
        return {
            'name': module.__class__.__name__,
            'kernel_size': module.kernel_size,
            'stride': module.stride,
            'padding': module.padding,
            'dilation': module.dilation,
            'return_indices': module.return_indices,
            'ceil_mode': module.ceil_mode,
        }
    
    def avgpool_config(self, module):
        assert isinstance(module, torch.nn.AvgPool2d), 'Not a AvgPool layer'
        return {
            'name': module.__class__.__name__,
            'kernel_size': module.kernel_size,
            'stride': module.stride,
            'padding': module.padding,
            'ceil_mode': module.ceil_mode,
            'count_include_pad': module.count_include_pad,
        }
    
    def dropout_config(self, module):
        assert isinstance(module, torch.nn.Dropout), 'Not a Dropout layer'
        return {
            'name': module.__class__.__name__,
            'p': module.p,
            'inplace': module.inplace,
        }
    
    def sequential_config(self, module):

        assert isinstance(module, torch.nn.Sequential), 'Not a Sequential layer'
        modules = []
        for submodule in module:
            modules.append(self.convert(submodule))
        return {
            'name': module.__class__.__name__,
            'modules': modules,
        }
    
    def module_to_config(self, module):
        if isinstance(module, torch.nn.Conv2d):
            return self.conv_config(module)
        elif isinstance(module, torch.nn.Linear):
            return self.linear_config(module)
        elif isinstance(module, torch.nn.BatchNorm2d):
            return self.batchnorm_config(module)
        elif isinstance(module, torch.nn.ReLU):
            return self.activation_config(module)
        elif isinstance(module, torch.nn.MaxPool2d):
            return self.maxpool_config(module)
        elif isinstance(module, torch.nn.AvgPool2d):
            return self.avgpool_config(module)
        elif isinstance(module, torch.nn.Dropout):
            return self.dropout_config(module)
        elif isinstance(module, torch.nn.Sequential):
            return self.sequential_config(module)
        else:
            raise ValueError('Unknown module type: {}'.format(module.__class__.__name__))
        
    def convert(self, module):
        return self.module_to_config(module)
    
    