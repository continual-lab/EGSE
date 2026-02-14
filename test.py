"""
Continual Learning Evaluation Utilities.

This module provides evaluation functionality for continual learning,
supporting task-specific module configurations and comprehensive metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional


class ContinualEvaluator:
    """
    Evaluator for continual learning with modular pathways.
    
    Handles evaluation across multiple tasks, tracking performance
    with task-specific module configurations and routing probabilities.
    """
    
    def __init__(
        self,
        config: Dict,
        verbose: bool = True
    ):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary containing:
                - device: Computation device
            verbose: Whether to print detailed information
        """
        self.config = config
        self.device = config['device']
        self.verbose = verbose
        
        # Storage for test loaders and results
        self.task_test_loaders = []
        self.task_best_acc = {i: 0 for i in range(10)}  # Best accuracy per task
        self.all_results = []
        self.best_overall_model_info = {}
    
    def add_task_test_loader(
        self,
        task_id: int,
        test_loader: torch.utils.data.DataLoader,
        task_name: Optional[str] = None
    ):
        """
        Add a test data loader for a task.
        
        Args:
            task_id: Task ID (0-indexed)
            test_loader: Test data loader
            task_name: Optional task name for logging
        """
        # Ensure list is long enough
        if task_id >= len(self.task_test_loaders):
            self.task_test_loaders.extend(
                [None] * (task_id + 1 - len(self.task_test_loaders))
            )
        
        self.task_test_loaders[task_id] = test_loader
        
        if self.verbose:
            name = task_name if task_name else f"Task {task_id}"
            print(f"Added test loader for {name}")
    
    def evaluate_all_tasks(
        self,
        task_models: List[nn.Module],
        routing_probs: Dict[str, List],
        current_task_id: int,
        epoch: int = -1
    ) -> tuple:
        """
        Evaluate all learned tasks with their specific configurations.
        
        Args:
            task_models: List of task-specific models
            routing_probs: Dictionary mapping task_id to routing probabilities
            current_task_id: Current task being trained
            epoch: Current training epoch
        
        Returns:
            Tuple of (task_accuracies, best_accuracies_dict)
        """
        results = {
            "epoch": epoch,
            "current_task_id": current_task_id,
            "task_results": {}
        }
        
        task_accuracies = []
        
        # Evaluate each learned task
        for task_id in range(current_task_id + 1):
            # Get task-specific model
            task_model = task_models[task_id]
            task_model.to(self.device)
            task_model.eval()
            
            # Get task-specific routing probabilities
            task_probs = routing_probs[f'{task_id}']
            
            # Evaluate
            acc = self._evaluate_single_task(task_model, task_probs, task_id)
            task_accuracies.append(acc)
            
            # Track results
            task_result = {
                "accuracy": acc,
                "is_new_best": False
            }
            
            # Update best accuracy
            if acc > self.task_best_acc.get(task_id, 0):
                self.task_best_acc[task_id] = acc
                task_result["is_new_best"] = True
            
            results["task_results"][task_id] = task_result
        
        # Compute summary metrics
        if task_accuracies:
            results["current_task_accuracy"] = task_accuracies[-1]
            results["average_accuracy"] = np.mean(task_accuracies)
            results["accuracy_std"] = np.std(task_accuracies) if len(task_accuracies) > 1 else 0
        
        # Record results
        self.all_results.append(results)
        
        return task_accuracies, self.task_best_acc
    
    def evaluate_single_task(
        self,
        model: nn.Module,
        routing_probs: List,
        task_id: int,
        epoch: int = -1
    ) -> float:
        """
        Evaluate model on a single task.
        
        Args:
            model: Task network
            routing_probs: Routing probabilities for this task
            task_id: Task ID
            epoch: Current epoch
        
        Returns:
            Accuracy percentage
        """
        model.eval()
        
        # Evaluate
        acc = self._evaluate_single_task(model, routing_probs, task_id)
        
        # Update best accuracy
        if acc > self.task_best_acc.get(task_id, 0):
            self.task_best_acc[task_id] = acc
        
        # Record result
        result = {
            "epoch": epoch,
            "task_id": task_id,
            "accuracy": acc
        }
        self.all_results.append(result)
        
        return acc
    
    def _evaluate_single_task(
        self,
        model: nn.Module,
        routing_probs: List,
        task_id: int
    ) -> float:
        """
        Internal method to evaluate on a single task.
        
        Args:
            model: Task network (already configured)
            routing_probs: Routing probabilities
            task_id: Task ID
        
        Returns:
            Accuracy percentage
        """
        test_loader = self.task_test_loaders[task_id]
        if test_loader is None:
            raise ValueError(f"No test loader available for task {task_id}")
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Handle different data formats
                if len(batch) == 2:
                    x, y = batch
                elif len(batch) >= 3:
                    x, y = batch[0], batch[1]
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)} elements")
                
                # Handle label formats
                if isinstance(y, (list, tuple)) and len(y) > 0:
                    y = y[0]
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                outputs = model(x, routing_probs)
                
                # Get predictions
                _, predicted = outputs.max(1)
                
                # Update statistics
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0
        return accuracy
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get evaluation summary statistics.
        
        Returns:
            Dictionary containing summary metrics
        """
        summary = {
            "num_tasks_evaluated": len(self.task_test_loaders),
            "best_accuracies": self.task_best_acc,
            "num_evaluations": len(self.all_results)
        }
        
        if self.all_results:
            # Get latest results
            latest = self.all_results[-1]
            if "average_accuracy" in latest:
                summary["latest_average_accuracy"] = latest["average_accuracy"]
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of evaluation results."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("Continual Learning Evaluation Summary")
        print("=" * 60)
        print(f"Tasks evaluated: {summary['num_tasks_evaluated']}")
        print(f"Total evaluations: {summary['num_evaluations']}")
        print("\nBest accuracies per task:")
        for task_id, acc in sorted(summary['best_accuracies'].items()):
            print(f"  Task {task_id}: {acc:.2f}%")
        
        if "latest_average_accuracy" in summary:
            print(f"\nLatest average accuracy: {summary['latest_average_accuracy']:.2f}%")
        print("=" * 60 + "\n")
