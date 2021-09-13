# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


from itertools import product
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # gates = [batch_size, num_experts]
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        # calculate num samples that each expert gets
        self._part_sizes = list((gates > 0).sum(0).cpu().numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, sf, of, ppf, pvf, gt_s=None, gt_o=None, gt_p_vec=None):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes

        sf_exp = sf[self._batch_index].squeeze(1)
        of_exp = of[self._batch_index].squeeze(1)
        ppf_exp = ppf[self._batch_index].squeeze(1)
        pvf_exp = pvf[self._batch_index].squeeze(1)

        gt_s_split = None
        gt_o_split = None
        gt_p_vec_split = None
        
        if gt_s != None:
          gt_s_exp = gt_s[self._batch_index].squeeze(0)
          gt_s_split = torch.split(gt_s_exp, self._part_sizes, dim=0)
        
        if gt_o != None:
          gt_o_exp = gt_o[self._batch_index].squeeze(0)
          gt_o_split = torch.split(gt_o_exp, self._part_sizes, dim=0)

        if gt_p_vec != None:
          gt_p_vec = gt_p_vec[self._batch_index].squeeze(0)
          gt_p_vec_split = torch.split(gt_p_vec, self._part_sizes, dim=0)
        

        return torch.split(sf_exp, self._part_sizes, dim=0), \
                torch.split(of_exp, self._part_sizes, dim=0), \
                torch.split(ppf_exp, self._part_sizes, dim=0), \
                torch.split(pvf_exp, self._part_sizes, dim=0), \
                gt_s_split, \
                gt_o_split,  \
                gt_p_vec_split 


    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """

        # Each expert outputs [s_score, o_score, p_score] and hence we 
        # first need to extract the output for each score.
        # once we have this, we can start combining the outputs.

        expert_s_out = [e[0] for e in expert_out]
        expert_o_out = [e[1] for e in expert_out]
        expert_p_out = [e[2] for e in expert_out]
        
        # apply exp to expert outputs, so we are not longer in log space
        stitched_s_score = torch.cat(expert_s_out, 0).exp()
        stitched_o_score = torch.cat(expert_o_out, 0).exp()
        stitched_p_score = torch.cat(expert_p_out, 0).exp()

        return self.combine_expert_output(stitched_s_score, expert_s_out, multiply_by_gates=multiply_by_gates), \
               self.combine_expert_output(stitched_o_score, expert_o_out, multiply_by_gates=multiply_by_gates), \
               self.combine_expert_output(stitched_p_score, expert_p_out, multiply_by_gates=multiply_by_gates)



    def combine_expert_output(self, stitched, expert_out, multiply_by_gates=True):
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)

        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True).to(self.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()


    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)




class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, model, input_size, num_experts, noisy_gating=True, k=4, object_num=None):
        super(MoE, self).__init__()
        self.object_num = object_num
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # instantiate experts
        self.experts = nn.ModuleList([model for _ in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True).to(self.device)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True).to(self.device)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0]).to(self.device), torch.tensor([1.0], ).to(self.device), validate_args=False)

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean()**2 + eps)


    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)




    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = (torch.arange(batch) * m + self.k).to(self.device)
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


    def noisy_top_k_gating(self, sf, of, ppf, pvf, train=True, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        # First concatenate all inputs along dim 1, so that the total size should be about [395, input_size]
        # This is a temporary measure to use the gating function as is, we will be playing with these values later.
        x = torch.cat((sf, of, ppf, pvf), 1)
        clean_logits = x @ self.w_gate
        if self.noisy_gating:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load



    def forward(self, sf, of, ppf, pvf, gt_s=None, gt_o=None, gt_p_vec=None, train=True, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(sf, of, ppf, pvf, train)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        sf_inputs, of_inputs, ppf_inputs, pvf_inputs, gt_s_inputs, gt_o_inputs, gt_p_vec_inputs = dispatcher.dispatch(sf, of, ppf, pvf, gt_s=gt_s, gt_o=gt_o, gt_p_vec=gt_p_vec)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](sf_inputs[i], of_inputs[i], ppf_inputs[i], pvf_inputs[i], gt_s_inputs[i], gt_o_inputs[i], gt_p_vec_inputs[i]) for i in range(self.num_experts)]
        s_score, o_score, p_score = dispatcher.combine(expert_outputs)
        
        return s_score, o_score, p_score, loss
    
    def infer_zero_shot_preference(self, strategy=None):
        pass
    
    def inference(self, sf, of, ppf, pvf):
        gates, _ = self.noisy_top_k_gating(sf, of, ppf, pvf, train=False)
        dispatcher = SparseDispatcher(self.num_experts, gates)

        sf_inputs, of_inputs, ppf_inputs, pvf_inputs, _, _ , _ = dispatcher.dispatch(sf, of, ppf, pvf, gt_s=None, gt_o=None, gt_p_vec=None)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i].inference(sf_inputs[i], of_inputs[i], ppf_inputs[i], pvf_inputs[i]) for i in range(self.num_experts)]
        s_prob, o_prob, p_prob = dispatcher.combine(expert_outputs) 

        return s_prob, o_prob, p_prob

    def predict(self, pairs, sf, of, ppf, pvf, trans_mat=None,
            inference_steps=2, inference_problistic=False,
            inference_object_conf_thres=0.1, inference_predicate_conf_thres=0.05):

        s_prob, o_prob, p_prob = self.inference(sf, of, ppf, pvf)

        obj_background_id = self.object_num-1
        s_max_idx = torch.argmax(s_prob, 1)
        o_max_idx = torch.argmax(o_prob, 1)
        valid_pair = (s_max_idx!=obj_background_id) & (o_max_idx!=obj_background_id)
        pairs = pairs[valid_pair]
        s_prob = s_prob[valid_pair, :-1]
        o_prob = o_prob[valid_pair, :-1]
        p_prob = p_prob[valid_pair]

        pairs = pairs.cpu().detach().numpy()
        s_prob = s_prob.cpu().detach().numpy()
        o_prob = o_prob.cpu().detach().numpy()
        p_prob = p_prob.cpu().detach().numpy()

        predictions = []
        for pair_id in range(len(pairs)):
            top_s_inds = np.where(s_prob[pair_id]>inference_object_conf_thres)[0]
            top_p_inds = np.where(p_prob[pair_id]>inference_predicate_conf_thres)[0]
            top_o_inds = np.where(o_prob[pair_id]>inference_object_conf_thres)[0]
            for s_class_id, p_class_id, o_class_id in product(top_s_inds, top_p_inds, top_o_inds):
                s_score = s_prob[pair_id, s_class_id]
                p_score = p_prob[pair_id, p_class_id]
                o_score = o_prob[pair_id, o_class_id]
                r_score = s_score*p_score*o_score
                sub_id, obj_id = pairs[pair_id]
                predictions.append({
                    'sub_id': sub_id,
                    'obj_id': obj_id,
                    'triplet': (s_class_id, p_class_id,o_class_id),
                    'score': r_score,
                    'triplet_scores': (s_score, p_score, o_score)
                })

        return predictions