from tinygrad import Tensor, nn, TinyJit
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
import numpy as np

def pixel_shuffle(x: Tensor, upscale_factor: int) -> Tensor:
    B, C, H, W = x.shape
    C_out = C // (upscale_factor * upscale_factor)
    x = x.reshape(B, upscale_factor, upscale_factor, C_out, H, W)
    x = x.permute(0, 3, 4, 1, 5, 2)
    return x.reshape(B, C_out, H * upscale_factor, W * upscale_factor)

class LeakyReLU:
    def __init__(self, negative_slope=0.2):
        self.negative_slope = negative_slope

    def __call__(self, x: Tensor) -> Tensor:
        return (x > 0).where(x, self.negative_slope * x)

class ResidualDenseBlock:
    def __init__(self, num_feat=64, num_grow_ch=32):
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, padding=1)
        self.lrelu = LeakyReLU(0.2)

    def __call__(self, x: Tensor) -> Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(Tensor.cat(x, x1, dim=1)))
        x3 = self.lrelu(self.conv3(Tensor.cat(x, x1, x2, dim=1)))
        x4 = self.lrelu(self.conv4(Tensor.cat(x, x1, x2, x3, dim=1)))
        x5 = self.conv5(Tensor.cat(x, x1, x2, x3, x4, dim=1))
        return x5 * 0.2 + x

class RRDB:
    def __init__(self, num_feat, num_grow_ch=32):
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def __call__(self, x: Tensor) -> Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet:
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, padding=1)
        self.body = [RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, padding=1)
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self._body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(_interpolate_2x(feat)))
        feat = self.lrelu(self.conv_up2(_interpolate_2x(feat)))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

    def _body(self, x: Tensor) -> Tensor:
        for block in self.body:
            x = block(x)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

def _interpolate_2x(x: Tensor) -> Tensor:
    return x.interpolate(size=(x.shape[2]*2, x.shape[3]*2), mode='nearest')

class RealESRGAN_x4plus:
    def __init__(self):
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=4,
            num_feat=64,
            num_block=23,
            num_grow_ch=32
        )
        self.jitted = None

    def __call__(self, x: Tensor) -> Tensor:
        return self.model(x)

    def get_jitted(self):
        if self.jitted is None:
            self.jitted = TinyJit(self.model)
        return self.jitted

    def load_weights(self, path: str):
        import torch
        state = torch.load(path, map_location='cpu')
        params = state['params_ema']
        state_dict = get_state_dict(self.model)
        for k in state_dict.keys():
            pytorch_key = k.replace('model.', '')
            if pytorch_key in params and isinstance(params[pytorch_key], torch.Tensor):
                state_dict[k].assign(params[pytorch_key].numpy())

    def save_weights(self, path: str):
        safe_save(get_state_dict(self.model), path)

def convert_pytorch_weights(pytorch_path: str, output_path: str):
    import torch
    state = torch.load(pytorch_path, map_location='cpu')
    params = state['params_ema']
    td = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            td[k] = v.numpy()
    safe_save(td, output_path)

if __name__ == "__main__":
    model = RealESRGAN_x4plus()
    x = Tensor.rand(1, 3, 64, 64)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")