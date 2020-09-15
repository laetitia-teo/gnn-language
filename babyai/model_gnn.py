import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import babyai.rl
from babyai.rl.utils.supervised_losses import required_heads
from gnns.models import SlotMemSparse2


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def scatter_sum(x, batch):
    nbatches = torch.max(batch).item() + 1
    nelems = len(batch)
    fx = x.shape[-1]
    i = torch.stack([batch, torch.arange(nelems).type(torch.LongTensor)])

    st = torch.sparse.FloatTensor(
        i,
        x,
        torch.Size([nbatches, nelems] + list(x.shape[1:])),
    )
    return torch.sparse.sum(st, dim=1).values()

class ACModelGNN(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=5, memory_dim=(4, 8), instr_dim=128, nheads=1,
                 use_instr=False, lang_model="gru", use_memory=False, arch="cnn1",
                 aux_info=None):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        # self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim

        self.obs_space = obs_space
        self.slot_memory_model = SlotMemSparse2(
            K=self.memory_dim[0],
            Fin=self.image_dim,
            Fmem=self.memory_dim[1],
            nheads=nheads
        )
        self.embedding_size = self.memory_dim[1]
        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    @property
    def memory_size(self):
        return self.memory_dim

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, obs_batch, m_batch, instr_embedding=None):
        # if self.use_instr and instr_embedding is None:
        #     instr_embedding = self._get_instr_embedding(obs.instr)
        # if self.use_instr and self.lang_model == "attgru":
        #     # outputs: B x L x D
        #     # memory: B x M
        #     mask = (obs.instr != 0).float()
        #     # The mask tensor has the same length as obs.instr, and
        #     # thus can be both shorter and longer than instr_embedding.
        #     # It can be longer if instr_embedding is computed
        #     # for a subbatch of obs.instr.
        #     # It can be shorter if obs.instr is a subbatch of
        #     # the batch that instr_embeddings was computed for.
        #     # Here, we make sure that mask and instr_embeddings
        #     # have equal length along dimension 1.
        #     mask = mask[:, :instr_embedding.shape[1]]
        #     instr_embedding = instr_embedding[:, :mask.shape[1]]
        #
        #     keys = self.memory2key(memory)
        #     pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
        #     attention = F.softmax(pre_softmax, dim=1)
        #     instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        output, memory = self.slot_memory_model(obs, memory, obs_batch, m_batch)

        embedding = scatter_sum(output, m_batch.type(torch.LongTensor))

        # if self.use_instr and not "filmcnn" in self.arch:
        #     embedding = torch.cat((embedding, instr_embedding), dim=1)

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        x = self.actor(embedding)

        dist = Categorical(logits=F.log_softmax(x))

        x = self.critic(embedding)
        value = x

        return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions}

    def _get_instr_embedding(self, instr):
        lengths = (instr != 0).sum(1).long()
        if self.lang_model == 'gru':
            out, _ = self.instr_rnn(self.word_embedding(instr))
            hidden = out[range(len(lengths)), lengths - 1, :]
            return hidden

        elif self.lang_model in ['bigru', 'attgru']:
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, final_states = self.instr_rnn(self.word_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            return outputs if self.lang_model == 'attgru' else final_states

        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))
