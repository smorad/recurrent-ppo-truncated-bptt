import torch
from torch import jit

from typing import Tuple, Union, List, Optional


class MaskedLSTM(jit.ScriptModule):
    """A masked LSTM that will only update hidden and cell
    states where mask == True"""

    def __init__(self, *cell_args, **cell_kwargs):
        super().__init__()
        self.cell = torch.nn.LSTMCell(*cell_args, **cell_kwargs)
        self.hidden_size = self.cell.hidden_size

    @jit.script_method
    def forward(
        self,
        input: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # shape: (B,L,*), [(1,B,*), (1,B,*)], (B,L)
        inputs = input.unbind(1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        if mask is None:
            mask = torch.ones(input.shape[:2], device=input.device)
        mask = mask.to(torch.bool).unsqueeze(-1).expand(-1, -1, self.hidden_size)
        if state[0].dim() == 3 and state[1].dim() == 3:
            state = (state[0].squeeze(), state[1].squeeze())
        h, c = state
        orig_shape = h.shape
        for i in range(len(inputs)):
            # Expects [B,*] input
            tmp_h, tmp_c = self.cell(inputs[i], (h,c))
            h = h.masked_scatter(mask[:, i], tmp_h)
            c = c.masked_scatter(mask[:, i], tmp_c)
            outputs += [tmp_h]
        return torch.stack(outputs, dim=1), (h.reshape(1,-1,self.hidden_size), c.reshape(1,-1,self.hidden_size))

    # mask.unsqueeze(-1).expand(1,2,8)
