import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

# -----------------------------------------------------------------------------
# 1. FNE (Fourier Number Embedding)
# -----------------------------------------------------------------------------
class FNE(nn.Module):
    def __init__(self, embedding_dim, int_digit_len=5, frac_digit_len=5, period_base_list=[2, 5], add_linear=False, device='cuda'):
        super().__init__()
        self.add_linear = add_linear
        self.period_list = self._get_period_list(period_base_list, minvalue=10**(-frac_digit_len), maxvalue=10**(int_digit_len+1)-1)
        self.period_base_list_len = len(period_base_list)
        
        with torch.no_grad():
            precomputed_cos_sin_matrix = self._create_precomputed_cos_sin_matrix(period_base_list).T
        self.register_buffer("precomputed_cos_sin_matrix", precomputed_cos_sin_matrix.to(device))
        
        period_tensor = torch.tensor(self.period_list, dtype=torch.float64, device=device)
        self.register_buffer("period_tensor", period_tensor)
        self.register_buffer("frequencies", (2 * torch.pi / period_tensor).to(device))
        
        self.embedding_dim = embedding_dim
        
        if self.add_linear:
            self.mlp = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim * 4),
                nn.GELU(),
                nn.Linear(self.embedding_dim * 4, self.embedding_dim)
            )
        else:
            self.mlp = None

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-5).to(device)
        self.device = device
        
        self.max_num_digits = int_digit_len + frac_digit_len
        powers_of_ten = torch.pow(10, torch.arange(self.max_num_digits, device=device)).long()[-(int_digit_len+frac_digit_len):]
        self.register_buffer("powers_of_ten", powers_of_ten)
        self.register_buffer("powers_of_ten_flipped", powers_of_ten.flip(0))

    def forward(self, number_scatter, len_gen=None):
        fourier_embedding = self.fourier_embedding(number_scatter, len_gen)
        if self.add_linear and self.mlp is not None:
            fourier_embedding = self.mlp(fourier_embedding)
        return fourier_embedding

    def fourier_embedding(self, number_scatter, len_gen=None):
        if len_gen is None: len_gen = 0
        flattened_number_scatter = number_scatter.flatten()
        number_in_hidden_space = self._turn_numbers_to_cosxsinx(flattened_number_scatter, len_gen)
        mask = (number_scatter != 0).unsqueeze(-1)
        masked_number_in_hidden_space = number_in_hidden_space.view(*number_scatter.shape, -1) * mask
        return masked_number_in_hidden_space

    def _get_period_list(self, period_base_list=[2, 5], minvalue=1e-5, maxvalue=1e5):
        period_list = set()
        current_value = 1.0
        while current_value >= minvalue:
            for base in period_base_list:
                period = current_value * base
                if period <= maxvalue: period_list.add(period)
            current_value /= 10
        current_value = 1
        while current_value <= maxvalue:
            for base in period_base_list:
                period = current_value * base
                if period <= maxvalue: period_list.add(period)
            current_value *= 10
        return sorted(period_list)

    def _turn_numbers_to_cosxsinx(self, numbers, len_gen=0):
        numbers = numbers.to(self.device, dtype=torch.float64) * (10**len_gen)
        cos_values = torch.cos(numbers.unsqueeze(1) * self.frequencies)
        sin_values = torch.sin(numbers.unsqueeze(1) * self.frequencies)
        cos_sin_stacked = torch.stack((cos_values, sin_values), dim=-1)
        cos_sin_interleaved = cos_sin_stacked.view(cos_values.size(0), -1)
        result = torch.zeros((len(numbers), self.embedding_dim), dtype=torch.float32, device=self.device)
        result[:, :cos_sin_interleaved.shape[1]] = cos_sin_interleaved
        return result

    def _create_precomputed_cos_sin_matrix(self, period_base_list):
        period_base_list = [base if type(base) != str else eval(base) for base in period_base_list]
        base_positions = torch.arange(10)
        cos_sin_list = []
        for period in period_base_list:
            w = 2 * torch.pi / period
            cos_sin_list.append(torch.cos(w * base_positions))
            cos_sin_list.append(torch.sin(w * base_positions))
        return torch.stack(cos_sin_list, dim=1)

    def fourier_compute_loss(self, before_decoder, label, int_digit_len, frac_digit_len, len_gen=None):
        if len_gen is None: len_gen = 0
        before_decoder = self.layer_norm(before_decoder)
        num_digits = int_digit_len + frac_digit_len
        num_cos_sin_per_digit = self.precomputed_cos_sin_matrix.shape[0]
        start_idx = 2 * self.period_base_list_len * len_gen
        end_idx = start_idx + (num_digits * num_cos_sin_per_digit)
        
        slices = before_decoder[..., start_idx:end_idx]
        slices = slices.view(-1, num_digits, num_cos_sin_per_digit)
        logits = torch.matmul(slices, self.precomputed_cos_sin_matrix)
        
        scaled_labels = (torch.round(label * (10 ** frac_digit_len))).long()
        digit_labels = (scaled_labels.unsqueeze(1) // self.powers_of_ten) % 10
        
        loss_per_digit = F.cross_entropy(logits.view(-1, 10), digit_labels.view(-1), reduction='none')
        return loss_per_digit.sum() / loss_per_digit.size(0)

    def fourier_compute_prediction(self, before_decoder, int_digit_len, frac_digit_len, len_gen=None):
        if len_gen is None: len_gen = 0
        before_decoder = self.layer_norm(before_decoder)
        batch_size = before_decoder.shape[0]
        num_cos_sin_per_digit = self.precomputed_cos_sin_matrix.shape[0]
        offset = 2 * self.period_base_list_len * len_gen
        
        predicted_number = torch.zeros(batch_size, dtype=torch.float64, device=before_decoder.device)
        
        for i in range(frac_digit_len):
            start = offset + i * num_cos_sin_per_digit
            end = start + num_cos_sin_per_digit
            logits = torch.matmul(before_decoder[..., start:end], self.precomputed_cos_sin_matrix)
            digit = torch.argmax(logits, dim=1).to(torch.float64)
            predicted_number += digit * (10 ** -(frac_digit_len - i))
            
        for j in range(int_digit_len):
            start = offset + (frac_digit_len + j) * num_cos_sin_per_digit
            end = start + num_cos_sin_per_digit
            logits = torch.matmul(before_decoder[..., start:end], self.precomputed_cos_sin_matrix)
            digit = torch.argmax(logits, dim=1).to(torch.float64)
            predicted_number += digit * (10 ** j)
            
        return predicted_number

# -----------------------------------------------------------------------------
# 2. XVAL
# -----------------------------------------------------------------------------
class XVAL(nn.Module):
    def __init__(self, embedding_dim, max_num=1000, device='cuda'):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.scale_factor = 5 / math.log(1 + max_num)
        self.num_head = nn.Linear(embedding_dim, 1).to(device)
        
    def forward(self, number_scatter, word_embedding):
        number_scatter = number_scatter.to(torch.float32)
        word_embedding = word_embedding.to(torch.float32)
        scaled_numbers = self.scale_factor * torch.log(1 + number_scatter.abs()) * torch.sign(number_scatter)
        mask = (scaled_numbers != 0).unsqueeze(-1)
        scaled_numbers_expanded = scaled_numbers.unsqueeze(-1).expand_as(word_embedding)
        return torch.where(mask, scaled_numbers_expanded * word_embedding, word_embedding)

    def compute_loss(self, before_decoder, label):
        label = label.to(torch.float32)
        scaled_label = self.scale_factor * torch.log(1 + label.abs()) * torch.sign(label)
        num_preds = self.num_head(before_decoder)
        return F.mse_loss(num_preds.squeeze(-1), scaled_label, reduction="mean")

    def compute_prediction(self, before_decoder):
        predictions = self.num_head(before_decoder).squeeze(-1)
        return (torch.exp(predictions.abs() / self.scale_factor) - 1) * torch.sign(predictions)

# -----------------------------------------------------------------------------
# 3. Vanilla Embedding
# -----------------------------------------------------------------------------
class VanillaEmbedding(nn.Module):
    def __init__(self, embedding_dim=1600, int_digit_len=5, frac_digit_len=5, device='cuda'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_num_digits = int_digit_len + frac_digit_len
        self.frac_digit_len = frac_digit_len
        self.device = device
        self.powers_of_ten = torch.pow(10, torch.arange(self.max_num_digits, device=device)).long()
        self.powers_of_ten_flipped = self.powers_of_ten

    def forward(self, number_scatter):
        flattened = number_scatter.flatten()
        vanilla_embeddings = self._turn_numbers_to_vanilla(flattened)
        padded = torch.zeros((len(flattened), self.embedding_dim), device=self.device)
        padded[:, :self.max_num_digits] = vanilla_embeddings
        padded = padded.view(*number_scatter.shape, self.embedding_dim)
        mask = (number_scatter != 0).unsqueeze(-1)
        return padded * mask

    def _turn_numbers_to_vanilla(self, numbers):
        scaled = (numbers * (10 ** self.frac_digit_len)).long()
        digits = (scaled.unsqueeze(1) // self.powers_of_ten) % 10
        return digits.float()

    def compute_loss(self, last_hidden_state, label):
        labels_digits = self._turn_numbers_to_vanilla(label).squeeze(1)
        predicted_digits = last_hidden_state[:, :self.max_num_digits]
        squared_errors = (predicted_digits - labels_digits) ** 2
        return torch.sqrt(torch.mean(squared_errors))

    def compute_prediction(self, last_hidden_state):
        predicted_digits = torch.round(last_hidden_state[:, :self.max_num_digits]).long() % 10
        return torch.sum(predicted_digits * self.powers_of_ten_flipped, dim=-1) / (10 ** self.frac_digit_len)

# -----------------------------------------------------------------------------
# 4. Gating Network (With Gumbel Support)
# -----------------------------------------------------------------------------
class GatingNetwork(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers=2, dropout=0.1, temperature=1.0, gating_type="sigmoid"):
        super().__init__()
        self.input_dim = hidden_dim + embedding_dim
        self.gating_type = gating_type  # required: store gating type
        self.temperature = temperature
        
        layers = []
        current_dim = self.input_dim
        for i in range(num_layers - 1):
            layers.extend([nn.Linear(current_dim, current_dim // 2), nn.GELU(), nn.Dropout(dropout)])
            current_dim = current_dim // 2
        
        final_layer = nn.Linear(current_dim, 1) 
        layers.append(final_layer)
        self.gating_network = nn.Sequential(*layers)

    def forward(self, hidden_states):
        logits = self.gating_network(hidden_states)
        
        if hasattr(self, 'gating_type') and self.gating_type == "gumbel":
            if self.training:
                # Gumbel-Sigmoid Sampling
                uniform_noise = torch.rand_like(logits)
                epsilon = 1e-10
                gumbel_noise = -torch.log(-torch.log(uniform_noise + epsilon) + epsilon)
                y_soft = torch.sigmoid((logits + gumbel_noise) / self.temperature)
                
                # Straight-Through Estimator
                y_hard = (y_soft > 0.5).float()
                return y_hard - y_soft.detach() + y_soft
            else:
                # Deterministic Hard Gating
                return (torch.sigmoid(logits) > 0.5).float()
        else:
            # Standard Sigmoid
            return torch.sigmoid(logits)

class HybridNumberEmbedding(nn.Module):
    def __init__(self, fne_encoder, standard_embedding, gating_network, num_token_id):
        super().__init__()
        self.fne_encoder = fne_encoder
        self.standard_embedding = standard_embedding
        self.gating_network = gating_network
        self.num_token_id = num_token_id
        self.embedding_dim = fne_encoder.embedding_dim
        
    def forward(self, hidden_states, numbers=None, number_indexs=None):
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        scatter_numbers = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        for batch_idx, (num_strs, pos_list) in enumerate(zip(numbers, number_indexs)):
            for num, pos in zip(num_strs, pos_list):
                if num is not None:
                    try:
                        val = float(num)
                        if not (torch.isnan(torch.tensor(val)) or torch.isinf(torch.tensor(val))):
                            scatter_numbers[batch_idx, pos] = val
                    except: continue
        
        numeric_embeds = self.fne_encoder(scatter_numbers)
        
        gating_inputs = []
        for batch_idx, idx_list in enumerate(number_indexs):
            valid_indices = [idx for idx in idx_list if 0 <= idx < seq_len]
            if not valid_indices: continue
            curr_hidden = hidden_states[batch_idx, valid_indices, :]
            curr_fne = numeric_embeds[batch_idx, valid_indices, :]
            gating_inputs.append(torch.cat([curr_hidden, curr_fne], dim=-1))
            
        if gating_inputs:
            final_feature_tensor = torch.cat(gating_inputs, dim=0)
        else:
            final_feature_tensor = torch.empty((0, hidden_states.shape[-1] + self.embedding_dim), device=device)
            
        gate_values = self.gating_network(final_feature_tensor)
        gate_ptr = 0
        for batch_idx, idx_list in enumerate(number_indexs):
            valid_indices = [idx for idx in idx_list if 0 <= idx < seq_len]
            if not valid_indices: continue
            num_pos = len(valid_indices)
            current_gates = gate_values[gate_ptr:gate_ptr + num_pos]
            gate_ptr += num_pos

            hidden_states[batch_idx, valid_indices, :] += (current_gates * numeric_embeds[batch_idx, valid_indices, :])
            # hidden_states[batch_idx, valid_indices, :] = numeric_embeds[batch_idx, valid_indices, :].to(hidden_states.dtype)

        return hidden_states, gate_values

class GatingLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    def forward(self, gate_values): return torch.tensor(0.0, device=gate_values.device)

class LoadBalancingLoss(nn.Module):
    def __init__(self, target_ratio=0.5, weight=1.0):
        super().__init__()
        self.weight = weight
    def forward(self, gate_values):
        if gate_values.numel() == 0: return torch.tensor(0.0, device=gate_values.device)
        return self.weight * torch.mean(gate_values ** 2)

class BinarizationLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    def forward(self, alpha):
        if alpha is None or alpha.numel() == 0: return torch.tensor(0.0, device=alpha.device)
        loss = alpha * (1 - alpha)
        return self.weight * loss.mean()