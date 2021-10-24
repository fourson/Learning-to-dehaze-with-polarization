import torch
import torch.jit


@torch.jit.script
def psnr(X, Y, data_range: float):
    """
    Peak Signal to Noise Ratio
    """

    mse = torch.mean((X - Y) ** 2)
    if mse == 0:
        return torch.tensor(50.)
    return 10 * torch.log10(data_range ** 2 / mse)


class PSNR(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'avg']

    def __init__(self, data_range=1., avg=True):
        super().__init__()
        self.data_range = data_range
        self.avg = avg
        self.__name__ = 'PSNR'

    @torch.jit.script_method
    def forward(self, X, Y):
        r = psnr(X, Y, self.data_range)
        if self.avg:
            return r.mean()
        else:
            return r
