import functools, tqdm
import torch
from torch import nn
import torch.optim as optim
import math

def rgetattr(obj, path):
    return functools.reduce(getattr, path.split("."), obj)

def project_orthogonal_subspace(vec, learned_vectors, normalization):
    U = learned_vectors.t() / normalization
    result = vec - U @ U.t() @ vec
    return result

class SteeredModel():
    def __init__(self, model, tokenizer, source_layer_idx=None, target_layer_idx=None, target_token_idxs=slice(None), layers_name=None, mlpout_name=None, normalization=1.0, vector_multiplier=1.0, num_steps=300, power=2, orthogonal_vectors=False, target_module="residual"):
        '''
        Note: this will mutate `model`
        '''
        self.model = model
        self.tokenizer = tokenizer
        
        # determine whether peft or not
        if "Peft" in type(model).__name__:
            self.peft = True
            self.base_model = model.base_model.model
        else:
            self.base_model = model
            self.peft = False
            
        # determine layers object
        if layers_name is None:
            if hasattr(self.base_model, "transformer"):  # gpt-2-like
                self.layers_name = "transformer.h"
            elif hasattr(self.base_model, "gpt_neox"): # pythia-like
                self.layers_name = "gpt_neox.layers"
            elif hasattr(self.base_model, "model"):  # mistral-like
                self.layers_name =  "model.model.layers"
            else:
                raise ValueError(f"don't know how to get layer list for {type(model)}")
        else:
            self.layers_name = layers_name
        self.layers = rgetattr(self.model, self.layers_name)
        
        # determine source layer
        if source_layer_idx is None:
            self.source_layer_idx = 7
        else:
            self.source_layer_idx = source_layer_idx
        
        # determine target layer
        if target_layer_idx is None:
            self.target_layer_idx = len(self.layers) - 8
        else:
            self.target_layer_idx = target_layer_idx
        
        # determine mlpout_name
        if mlpout_name is None:
            if "QWen" in type(self.base_model).__name__:
                self.mlpout_name = "mlp.c_proj"
            elif hasattr(self.base_model, "gpt_neox"):
                self.mlpout_name = "mlp.dense_4h_to_h"
            else:
                self.mlpout_name = "mlp.down_proj" # otherwise guess "down_proj"
        else:
            self.mlpout_name = mlpout_name
            
        if self.peft:
            self.mlpout_name += ".lora_B.default"
        # get width
        self.width = rgetattr(self.layers[0], self.mlpout_name).out_features
        
        # set other hyper-parameters
        self.normalization = normalization
        self.vector_multiplier = vector_multiplier
        self.target_token_idxs = target_token_idxs
        self.num_steps = num_steps
        self.power = power
        self.orthogonal_vectors = orthogonal_vectors
        self.target_module = target_module

        # don't need to store grads for most parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # set bias
        rgetattr(rgetattr(self.model, f"{self.layers_name}")[self.source_layer_idx], self.mlpout_name).bias = nn.Parameter(
            torch.zeros(self.width, device=self.model.device)
        )
        self.bias = rgetattr(rgetattr(self.model, f"{self.layers_name}")[self.source_layer_idx], self.mlpout_name).bias
        pass
    
    
    def train(self, examples, num_vectors):
        self.num_vectors = num_vectors
        self.learned_vectors = torch.zeros(self.num_vectors, self.width, device=self.model.device)

        num_steps = self.num_steps
        orthogonal_vectors = self.orthogonal_vectors
        normalization = self.normalization
        power = self.power
        
        # compute unsteered targets
        self.zero_steering_vector()
        self.unsteered_targets = []
        for i in range(len(examples)):
            model_inputs = self.tokenizer([examples[i]], return_tensors="pt", padding=False).to(self.model.device)
            with torch.no_grad():
                if self.target_module == "residual":
                    hidden_states = self.model(model_inputs["input_ids"], output_hidden_states=True).hidden_states
                elif self.target_module == "attn":
                    hidden_states = self.model(model_inputs["input_ids"], output_attentions=True).attentions
                else:
                    raise ValueError("target_module must be 'residual' or 'attn'")
                self.unsteered_targets.append(hidden_states[self.target_layer_idx][:, self.target_token_idxs, :])

        
        # loop over vectors
        losses_all = []
        bias = self.bias
        for i in (pbar := tqdm.tqdm(range(num_vectors))):
            
            # initialize
            losses = []
            with torch.no_grad():
                if self.orthogonal_vectors:
                    bias.data = normalization*nn.functional.normalize(
                        project_orthogonal_subspace(
                            torch.randn(self.width, device="cuda"), self.learned_vectors, self.normalization
                        ), 
                        dim=0
                    )
                else:
                    bias.data = normalization*nn.functional.normalize(torch.randn(self.width, device="cuda"), dim=0)
                        
            # optimizer
            optimizer = optim.AdamW([bias],
                                    lr=.001, betas=(.9,.98), weight_decay=0.0, amsgrad=True)
            
            # training loop
            for t in range(num_steps):
                
                # compute gradient
                optimizer.zero_grad()
                for s in range(len(examples)):
                    model_inputs = self.tokenizer([examples[s]], return_tensors="pt", padding=False).to(self.model.device)
    
                    # compute steered target
                    if self.target_module == "residual":
                        hidden_states = self.model(model_inputs["input_ids"], output_hidden_states=True).hidden_states
                    elif self.target_module == "attn":
                        hidden_states = self.model(model_inputs["input_ids"], output_attentions=True).attentions
                    else:
                        raise ValueError("target_module must be 'residual' or 'attn'")
                    target = hidden_states[self.target_layer_idx][:, self.target_token_idxs, :]
                    if power == "max":
                        loss = -(target-self.unsteered_targets[s]).norm(dim=1).pow(2).max()
                    else:
                        loss = -(target-self.unsteered_targets[s]).norm(dim=1).pow(power).sum().pow(1/power)
                    loss.backward()
                
                # project gradient to subspace orthogonal to previous learned vectors (if orthogonal_vectors is True)
                if orthogonal_vectors:
                    with torch.no_grad():
                        bias.grad = project_orthogonal_subspace(
                            bias.grad, 
                            self.learned_vectors, 
                            normalization
                        )
                
                # project gradient to tangent space of sphere
                with torch.no_grad():
                    bias.grad -= torch.dot(
                        bias.grad, bias
                    ) * bias / (normalization**2)

                # step
                optimizer.step()

                # project steering vector to subspace orthogonal to previous learned vectors (if orthogonal_vectors is True)
                if orthogonal_vectors:
                    with torch.no_grad():
                        bias.data = project_orthogonal_subspace(bias, self.learned_vectors, normalization)

                # normalize
                with torch.no_grad():
                    bias.data = nn.functional.normalize(bias.data, dim=0) * normalization
                    
                with torch.no_grad():
                    l_ = loss.detach().item()
                losses.append(l_)
            
            with torch.no_grad():
                self.learned_vectors[i,:] = bias.data.detach()
            losses_all.append(losses)
            
        self.losses_all = losses_all
        pass

    def set_steering_vector(self, i):
        with torch.no_grad():
            self.bias.data = self.learned_vectors[i,:] * self.vector_multiplier
        pass
    
    def zero_steering_vector(self):
        if self.bias is not None:
            with torch.no_grad():
                self.bias.data *= 0.0
        pass
