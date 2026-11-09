import torch

from ultralytics.nn.modules import FGBSCA, FLPSPD, LHMSP, P2ASFF


def test_pfgr_module_shapes():
    x = torch.randn(2, 16, 32, 32)
    assert FLPSPD(16, 32)(x).shape == (2, 32, 16, 16)
    assert LHMSP(16, 32)(x).shape == (2, 32, 32, 32)
    assert FGBSCA(16, 16)(x).shape == x.shape

    fused = torch.randn(2, 7 * 16, 32, 32)
    assert P2ASFF(7 * 16, 16)(fused).shape == (2, 16, 32, 32)
