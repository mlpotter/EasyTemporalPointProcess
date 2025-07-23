from logging import config
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.model.torch_model.torch_baselayer import TimeShiftedPositionalEncoding

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
        self.use_ln = model_config.use_ln
        # transform inter-event time embedding
        self.layer_dense_1 = nn.Linear(in_features=1, out_features=self.hidden_size)

        # concat rnn states and inter-event time embedding
        self.layer_dense_2 = nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size)

        # mlp layers
        self.module_list = nn.ModuleList(
            [nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size) for _ in
            range(self.num_mlp_layers - 1)])

        self.layer_dense_3 = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                                    out_features=self.num_event_types),
                                        nn.Softplus())

        self.params_eps = torch.finfo(torch.float32).eps  # ensure positiveness of parameters

        # self.bias_param = nn.Parameter(torch.ones(self.num_event_types,))

        self.init_weights_positive()

    def init_weights_positive(self):
        for p in self.parameters():
            p.data = torch.abs(p.data)
            p.data = torch.clamp(p.data, min=self.params_eps)

    def forward(self, hidden_states, time_delta_seqs):
        # print(hidden_states.shape, time_delta_seqs.shape)
        for p in self.parameters():
            p.data = torch.clamp(p.data, min=self.params_eps)

        time_delta_seqs.requires_grad_(True)

        # [batch_size, seq_len, hidden_size]
        t = self.layer_dense_1(time_delta_seqs.unsqueeze(dim=-1))  # [batch_size, seq_len, hidden_size]
        # t = self.layer_dense_1(torch.log(time_delta_seqs.unsqueeze(dim=-1) + 1e-16)) if self.use_ln else self.layer_dense_1(time_delta_seqs.unsqueeze(dim=-1))  # [batch_size, seq_len, hidden_size]
        
        # t = time_delta_seqs.unsqueeze(dim=-1)

        # [batch_size, seq_len, hidden_size]
        out = torch.tanh(self.layer_dense_2(torch.cat([hidden_states, t], dim=-1)))
        for layer in self.module_list:
            out = torch.tanh(layer(out))

        # [batch_size, seq_len, num_event_types]
        integral_lambda = self.layer_dense_3(out) #+ F.softplus(self.bias_param * time_delta_seqs.unsqueeze(dim=-1))
        # print(integral_lambda.shape, time_delta_seqs.shape)
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
                integral_lambda.sum(dim=-1).mean(),
                time_delta_seqs,
                create_graph=True, retain_graph=True)[0]
            derivative_integral_lambda = derivative_integral_lambda.unsqueeze(-1).expand(*derivative_integral_lambda.shape, self.num_event_types) / self.num_event_types

        return integral_lambda, derivative_integral_lambda


class FullyNN(TorchBaseModel):
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
        super(FullyNN, self).__init__(model_config)

        self.rnn_type = model_config.rnn_type
        self.rnn_list = [nn.LSTM, nn.RNN, nn.GRU]
        self.n_layers = model_config.num_layers
        self.dropout_rate = model_config.dropout_rate
        # for sub_rnn_class in self.rnn_list:
        #     if sub_rnn_class.__name__ == self.rnn_type:
        #         self.layer_rnn = sub_rnn_class(input_size=1 ,#+ self.hidden_size,
        #                                     hidden_size=self.hidden_size,
        #                                     num_layers=self.n_layers,
        #                                     batch_first=True,
        #                                     dropout=self.dropout_rate)
        embed_dim = model_config.hidden_size
        if embed_dim % model_config.num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({model_config.num_heads})")
        self.layer_rnn = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=model_config.num_heads,
            dim_feedforward=embed_dim,
            dropout=model_config.dropout_rate,
            batch_first=True,
            ),
            num_layers=model_config.num_layers
        )
        self.layer_input_projection = nn.Linear(1, embed_dim)
        self.layer_position_emb = TimeShiftedPositionalEncoding(
            d_model=embed_dim
        )
        self.layer_intensity = CumulHazardFunctionNetwork(model_config)
        self.use_ln = model_config.use_ln
        self.layer_type_emb = None
        self.num_heads = model_config.num_heads

    def forward(self, time_seqs, time_delta_seqs, type_seqs, attention_mask_seqs):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.

        Returns:
            tensor: hidden states at event times.
        """
        # [batch_size, seq_len, hidden_size]
        # type_embedding = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size + 1]
        # rnn_input = torch.cat((type_embedding, 
        #                        time_delta_seqs.unsqueeze(-1), 
        #                     #    time_seqs.unsqueeze(-1)
        #                        ), dim=-1)
        
        
        time_delta_seqs_log = torch.log(time_delta_seqs.unsqueeze(-1) + 1e-16) if self.use_ln else time_delta_seqs.unsqueeze(-1)

        time_seqs_log = torch.log(time_seqs.unsqueeze(-1) + 1e-16) if self.use_ln else time_seqs.unsqueeze(-1)

        input_embed = self.layer_input_projection(time_delta_seqs_log)  # [batch_size, seq_len, hidden_size * num_heads]

        position_embed = self.layer_position_emb(time_seqs_log.squeeze(-1),time_delta_seqs_log.squeeze(-1))

        rnn_input = input_embed + position_embed  # [batch_size, seq_len, hidden_size * num_heads]
        # [batch_size, seq_len, hidden_size]
        # states right after the event
        hidden_states = self.layer_rnn(rnn_input,mask=attention_mask_seqs.repeat_interleave(self.num_heads, dim=0))

        return hidden_states  # remove the first time step (hidden node), as it is not used in the model

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            list: loglike loss, num events.
        """
        # [batch_size, seq_len]
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, batch_attention_mask = batch

        # [batch_size, seq_len, hidden_size]
        # hidden_states = self.forward(
        #     time_seqs[:, :-1],
        #     time_delta_seqs[:, :-1],
        #     type_seqs[:, :-1],
        # )

        # [batch_size, seq_len, hidden_size]
        hidden_states = self.forward(
            time_seqs=time_seqs[:, :-1],
            time_delta_seqs=time_delta_seqs[:, :-1],
            type_seqs=type_seqs[:, :-1],
            attention_mask_seqs=batch_attention_mask[:,:-1,:-1]
        )
        
        # print(hidden_states)
        # print(hidden_states.shape)
        # [batch_size, seq_len, num_event_types]
        integral_lambda, derivative_integral_lambda = self.layer_intensity(hidden_states, time_delta_seqs[:, 1:])

        # First, add an epsilon to every marked intensity for stability
        derivative_integral_lambda += self.eps

        # Compute components for each LL term
        log_marked_event_lambdas = derivative_integral_lambda.log()

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
