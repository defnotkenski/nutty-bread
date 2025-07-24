import torch
from torch.func import vmap, stack_module_state, functional_call
from custom.models.saint_transformer.config import SAINTConfig


class VMapEnsembleTrainer:
    def __init__(self, model_factory, num_models: int, seeds: list[int]):
        self.num_models = num_models
        self.seeds = seeds or [777 + i for i in range(num_models)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create models with different seeds
        self.models = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            model = model_factory()
            self.models.append(model)

        # Stack parameters for vmap
        self.params, self.buffers = stack_module_state(self.models)
        self.base_model = self.models[0]

        # Move to device
        self.params = {k: v.to(self.device) for k, v in self.params.items()}
        self.buffers = {k: v.to(self.device) for k, v in self.buffers.items()}

    def vectorized_forward(self, batch):
        x, y, attention_mask = batch

        def single_model_setup(params, buffers):
            model_output = functional_call(self.base_model, (params, buffers), (x, y, attention_mask, True))
            return model_output

        # Vectorize across model dimension (0)
        return vmap(single_model_setup, in_dims=(0, 0))(self.params, self.buffers)

    def train_step(self, batch, optimizers):
        losses, probs, y_masked = self.vectorized_forward(batch)

        for i, (loss, optimizer) in enumerate(zip(losses, optimizers)):
            optimizer.zero_grad()
            retain_graph = i < len(losses) - 1
            loss.backward(retain_graph=retain_graph)
            optimizer.step()

        return losses.mean().item()

    def create_optimizers(self, config: SAINTConfig):
        optimizers = []
        for i in range(self.num_models):
            optimizer = torch.optim.AdamW(
                self.models[i].parameters(),
                lr=config.learning_rate,
                betas=config.betas,
                weight_decay=config.weight_decay,
            )

            optimizers.append(optimizer)

        return optimizers

    def get_averaged_weights(self):
        averaged = {}
        for name, param_stack in self.params.items():
            averaged[name] = param_stack.mean(dim=0)

        return averaged

    def create_final_model(self, model_factory):
        torch.manual_seed(777)
        final_model = model_factory()
        final_model.load_state_dict(self.get_averaged_weights())

        return final_model
