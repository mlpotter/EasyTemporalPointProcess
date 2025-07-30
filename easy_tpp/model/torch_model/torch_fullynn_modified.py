import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class CumulHazardFunctionNetwork(nn.Module):
    """Cumulative Hazard Function Network
    ref: https://github.com/wassname/torch-neuralpointprocess
    """

    def __init__(self, model_config):
        super(CumulHazardFunctionNetwork, self).__init__()
        self.hidden_size = model_config.hidden_size
        self.num_mlp_layers = model_config.model_specs['num_mlp_layers']
        self.num_event_types = model_config.num_event_types
        self.proper_marked_intensities = model_config.model_specs["proper_marked_intensities"]

        # transform inter-event time embedding
        self.layer_dense_1 = nn.Linear(in_features=1, out_features=self.hidden_size)

        # concat rnn states and inter-event time embedding
        self.layer_dense_2 = nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size)

        # mlp layers
        self.module_list = nn.ModuleList(
            [nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size) for _ in
             range(self.num_mlp_layers - 1)])

        self.layer_dense_3 = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                                     out_features=self.num_event_types),
                                           nn.Softplus())

        self.bias_term = nn.Linear(in_features=model_config.hidden_size,out_features=1,bias=True)

        self.params_eps = torch.finfo(torch.float32).eps  # ensure positiveness of parameters

        self.init_weights_positive()

    def init_weights_positive(self):
        for p in self.parameters():
            p.data = torch.abs(p.data)
            p.data = torch.clamp(p.data, min=self.params_eps)

    def forward(self, hidden_states, time_delta_seqs):
        for p in self.parameters():
            p.data = torch.clamp(p.data, min=self.params_eps)

        time_delta_seqs.requires_grad_(True)

        # [batch_size, seq_len, hidden_size]
        t = self.layer_dense_1(time_delta_seqs.unsqueeze(dim=-1))

        # [batch_size, seq_len, hidden_size]
        out = torch.tanh(self.layer_dense_2(torch.cat([hidden_states, t], dim=-1)))
        for layer in self.module_list:
            out = torch.tanh(layer(out))

        t_base = self.layer_dense_1(torch.zeros_like(time_delta_seqs.unsqueeze(dim=-1),device=time_delta_seqs.device))  # [batch_size, seq_len, hidden_size]

        # [batch_size, seq_len, hidden_size]
        out_base = torch.tanh(self.layer_dense_2(torch.cat([hidden_states, t_base], dim=-1)))
        for layer in self.module_list:
            out_base = torch.tanh(layer(out_base))

        out_bias = F.relu(self.bias_term(hidden_states))  # [batch_size, seq_len, 1]


        # [batch_size, seq_len, num_event_types]
        integral_lambda = self.layer_dense_3(out) - self.layer_dense_3(out_base) + out_bias * time_delta_seqs.unsqueeze(dim=-1)

        # [batch_size, seq_len, num_event_types]
        if self.proper_marked_intensities:
            derivative_integral_lambdas = []
            for i in range(integral_lambda.shape[-1]):  # iterate over marks
                derivative_integral_lambdas.append(grad(
                    integral_lambda[..., i].mean(),
                    time_delta_seqs,
                    create_graph=True, retain_graph=True)[0])
            derivative_integral_lambda = torch.stack(derivative_integral_lambdas, dim=-1)  # TODO: Check that it is okay to iterate over marks like this
        else:
            derivative_integral_lambda = grad(
                integral_lambda.sum(),
                time_delta_seqs,
                create_graph=True, retain_graph=True)[0]
            derivative_integral_lambda = derivative_integral_lambda.unsqueeze(-1).expand(*derivative_integral_lambda.shape, self.num_event_types) / self.num_event_types

        return integral_lambda, derivative_integral_lambda


class FullyNNModified(TorchBaseModel):
    """Torch implementation of
        Fully Neural Network based Model for General Temporal Point Processes, NeurIPS 2019.
        https://arxiv.org/abs/1905.09690

        ref: https://github.com/KanghoonYoon/torch-neuralpointprocess/blob/master/module.py;
            https://github.com/wassname/torch-neuralpointprocess
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(FullyNNModified, self).__init__(model_config)

        self.rnn_type = model_config.rnn_type
        self.rnn_list = [nn.LSTM, nn.RNN, nn.GRU]
        self.n_layers = model_config.num_layers
        self.dropout_rate = model_config.dropout_rate
        for sub_rnn_class in self.rnn_list:
            if sub_rnn_class.__name__ == self.rnn_type:
                self.layer_rnn = sub_rnn_class(input_size=1,
                                                hidden_size=self.hidden_size,
                                                num_layers=self.n_layers,
                                                batch_first=True,
                                                dropout=self.dropout_rate)

        self.layer_intensity = CumulHazardFunctionNetwork(model_config)
        self.use_ln = model_config.use_ln
        self.right_censoring = model_config.right_censoring

    def forward(self, time_seqs, time_delta_seqs, type_seqs):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.

        Returns:
            tensor: hidden states at event times.
        """
        # [batch_size, seq_len, hidden_size]

        time_delta_seqs_log = torch.log(time_delta_seqs.unsqueeze(-1) + 1e-6) if self.use_ln else time_delta_seqs.unsqueeze(-1)

        time_seqs_log = torch.log(time_seqs.unsqueeze(-1) + 1e-6) if self.use_ln else time_seqs.unsqueeze(-1)

        # [batch_size, seq_len, hidden_size + 1]
        rnn_input = time_delta_seqs_log

        # [batch_size, seq_len, hidden_size]
        # states right after the event
        hidden_states, _ = self.layer_rnn(rnn_input)

        return hidden_states

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            list: loglike loss, num events.
        """
        # [batch_size, seq_len]
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ , batch_len_seqs = batch

        # [batch_size, seq_len, hidden_size]
        hidden_states = self.forward(
            time_seqs[:, :-1],
            time_delta_seqs[:, :-1],
            type_seqs[:, :-1],
        )
        # [batch_size, seq_len, num_event_types]
        integral_lambda, derivative_integral_lambda = self.layer_intensity(hidden_states, time_delta_seqs[:, 1:])

        # First, add an epsilon to every marked intensity for stability
        derivative_integral_lambda += self.eps

        # Compute components for each LL term
        log_marked_event_lambdas = derivative_integral_lambda.log()

        # If right censoring is used, we need to handle the last event in nll_loss
        if self.right_censoring:
            type_seqs[torch.arange(len(batch_len_seqs)), batch_len_seqs-1] = self.pad_token_id

        # Compute event LL - [batch_size, seq_len]
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),  # mark dimension needs to come second, not third to match nll_loss specs
            target=type_seqs[:, 1:],
            ignore_index=self.pad_token_id,  # Padded events have a pad_token_id as a value
            reduction='none', # Does not aggregate, and replaces what would have been the log(marked intensity) with 0.
        )

        # [batch_size, seq_len]
        # multiplied by sequence mask
        non_event_ll = integral_lambda.sum(-1) * batch_non_pad_mask[:, 1:]
        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]
        loss = - (event_ll - non_event_ll).sum()

        return loss, num_events

    def compute_intensities_at_sample_times(self,
                                            time_seqs,
                                            time_delta_seqs,
                                            type_seqs,
                                            sample_dtimes,
                                            **kwargs):
        """Compute hidden states at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types], intensity at all sampled times.
        """

        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        # [batch_size, seq_len, hidden_size]
        hidden_states = self.forward(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
        )

        num_samples = sample_dtimes.size()[-1]
        batch_size, seq_len, hidden_size = hidden_states.shape

        hidden_states_ = hidden_states[..., None, :].expand(batch_size, seq_len, num_samples, hidden_size)
        _, derivative_integral_lambda = self.layer_intensity.forward(
            hidden_states=hidden_states_,
            time_delta_seqs=sample_dtimes,
        )

        if compute_last_step_only:
            lambdas = derivative_integral_lambda[:, -1:, :, :]
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = derivative_integral_lambda
        return lambdas