"""
Proximal Policy Optimization (PPO) Trainer.

This module implements the PPO algorithm for training the policy network
to discover optimal module compositions.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from transformers import get_linear_schedule_with_warmup
from typing import Dict


class PPOTrainer:
    """
    PPO trainer for policy optimization.
    
    Implements the Proximal Policy Optimization algorithm with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Entropy regularization
    - Value function learning
    """
    
    def __init__(self, policy_net, config: Dict):
        """
        Initialize PPO trainer.
        
        Args:
            policy_net: Policy network to train
            config: Configuration dictionary containing:
                - lr: Learning rate
                - clip_eps: PPO clipping epsilon
                - entropy_coef: Entropy coefficient
                - discount: Discount factor (gamma)
                - gae_tau: GAE lambda parameter
                - ppo_epochs: Number of PPO update epochs
                - rl_epochs: Total RL training epochs
                - repeat: Batch size (number of trajectories)
                - steps: Trajectory length
                - device: Computation device
        """
        self.config = config
        self.policy_net = policy_net
        self.optimizer = torch.optim.Adam(
            policy_net.parameters(), 
            lr=config['lr']
        )
        self.clip_eps = config['clip_eps']
        self.entropy_coef = config['entropy_coef']
        
        # Learning rate scheduler with warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * config['rl_epochs'] * config['ppo_epochs']),
            num_training_steps=config['rl_epochs'] * config['ppo_epochs']
        )
    
    def compute_advantages(self, storage):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            storage: Storage object containing trajectory data
        """
        values = storage.v
        
        # Initialize returns and advantages
        ret = values[-1].detach()
        advantages = torch.zeros(
            (self.config['repeat'], 1), 
            device=self.config['device']
        )
        all_returns = torch.zeros(
            (self.config['repeat'], self.config['steps']), 
            device=self.config['device']
        )
        all_advantages = torch.zeros(
            (self.config['repeat'], self.config['steps']), 
            device=self.config['device']
        )
        
        # Backward computation of advantages and returns
        for i in reversed(range(self.config['steps'])):
            # Compute returns
            ret = (
                storage.r[i] + 
                self.config['discount'] * (1 - storage.m[i]) * ret
            )
            
            # Compute TD error
            td_error = (
                storage.r[i] + 
                self.config['discount'] * (1 - storage.m[i]) * values[i + 1] - 
                values[i]
            )
            
            # Update advantages with GAE
            advantages = (
                advantages * self.config['gae_tau'] * 
                self.config['discount'] * (1 - storage.m[i]) + 
                td_error
            )
            
            # Store
            all_advantages[:, i:i+1] = advantages.detach()
            all_returns[:, i:i+1] = ret.detach()
        
        # Add to storage
        for i in range(self.config['steps']):
            storage.add({
                'adv': all_advantages[:, i:i+1],
                'ret': all_returns[:, i:i+1]
            })
    
    def update_policy(self, storage, progress: float):
        """
        Update policy using PPO.
        
        Args:
            storage: Storage containing trajectory data
            progress: Training progress (0-1) for annealing
            
        Returns:
            Tuple of (policy_loss, value_loss)
        """
        print(f"Training progress: {progress:.2%}")
        
        # Compute advantages
        self.compute_advantages(storage)
        
        # Load trajectory data
        states, actions, params, log_probs_old, returns, advantages = \
            storage.load(['s', 'a', 'p', 'log_probs', 'ret', 'adv'])
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        
        # Multiple epochs of PPO updates
        for _ in range(self.config['ppo_epochs']):
            # Random shuffling for mini-batch training
            shuffled_indices = torch.randperm(states.size(0))
            
            # Process in mini-batches
            batch_size = max(1, int(states.size(0) / 16))
            for start_idx in range(0, states.size(0), batch_size):
                end_idx = min(start_idx + batch_size, states.size(0))
                batch_indices = shuffled_indices[start_idx:end_idx]
                
                # Forward pass
                act_logits, param_mean, std, value = self.policy_net(
                    states[batch_indices]
                )
                
                # Action distribution
                action_dist = torch.distributions.Independent(
                    torch.distributions.Categorical(logits=act_logits),
                    reinterpreted_batch_ndims=1
                )
                
                # Parameter distribution (routing probabilities)
                selected_params = torch.gather(
                    param_mean, -1, actions[batch_indices].unsqueeze(-1)
                ).squeeze(-1)
                
                # Adaptive standard deviation (annealed with progress)
                adaptive_std = std * (1 - progress)
                param_dist = torch.distributions.Normal(selected_params, adaptive_std)
                
                # Compute log probabilities
                log_pi_a = (
                    action_dist.log_prob(actions[batch_indices]) +
                    param_dist.log_prob(params[batch_indices]).sum(-1)
                )
                
                # Entropy for exploration
                entropy = action_dist.entropy()
                
                # PPO ratio
                ratio = torch.exp(log_pi_a - log_probs_old[batch_indices])
                
                # Clipped surrogate objective
                obj = ratio * advantages[batch_indices]
                obj_clipped = (
                    ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * 
                    advantages[batch_indices]
                )
                
                # Policy loss with entropy regularization
                policy_loss = -(torch.min(obj, obj_clipped)).mean() - (
                    self.entropy_coef * (1 - progress) * entropy
                ).mean()
                
                # Value loss
                value_loss = 0.5 * (value - returns[batch_indices]).pow(2).mean()
                
                # Combined loss
                total_loss = policy_loss + value_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
            
            # Update learning rate
            self.scheduler.step()
        
        # Return average losses
        avg_policy_loss = total_policy_loss / self.config['ppo_epochs']
        avg_value_loss = total_value_loss / self.config['ppo_epochs']
        
        return avg_policy_loss, avg_value_loss
    
    def save_model(self, epoch: int, path: str = "model.pth"):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            path: Path to save checkpoint
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str = "model.pth"):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Model loaded from epoch {epoch}")
        return epoch
