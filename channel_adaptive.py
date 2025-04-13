# %%
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################
# 1. FixedCosBasis: 고정 Fourier Basis 생성
#############################################


class FourierBasisGenerator(nn.Module):
    """
    FourierBasisGenerator는 입력 in_features 차원에 대해,
    high_freq_num, low_freq_num, phi_num을 이용하여 여러 phase를 가진 cosine basis들을 생성합니다.

    출력: 
      - 기본적으로 : [((high_freq_num + low_freq_num)*phi_num), in_features] 모양
      - select_two=True 인 경우: [2, in_features], 
            첫 번째 row는 낮은 주파수(low_freq 그룹의 첫 번째, phi=0에 가까움),
            두 번째 row는 높은 주파수(high_freq 그룹의 마지막, phi=마지막 값)로 설정합니다.
    """

    def __init__(self, low_rank: int, high_freq_num: int, low_freq_num: int, phi_num: int, alpha: float = 0.05, select_two: bool = False):
        super(FourierBasisGenerator, self).__init__()
        self.low_rank = low_rank
        self.high_freq_num = high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha = alpha
        self.select_two = select_two
        self.bases = self.init_bases()

    def init_bases(self):
        # Phase offsets: phi_set ∈ ℝ^(phi_num)
        phi_set = np.array(
            [2 * math.pi * i / self.phi_num for i in range(self.phi_num)])

        # low_freq: 예: [(i+1)/low_freq_num for i in range(low_freq_num)]
        low_freq = np.array(
            [(i + 1) / self.low_freq_num for i in range(self.low_freq_num)])
        # high_freq: 예: [i+1 for i in range(high_freq_num)]
        high_freq = np.array([i + 1 for i in range(self.high_freq_num)])

        # T_max: low_freq 그룹이 존재하면 T_max = 2π / (첫 번째 low_freq), 아니면 2π / (min(high_freq))
        if len(low_freq) != 0:
            T_max = 2 * math.pi / low_freq[0]
        else:
            T_max = 2 * math.pi / min(high_freq)

        # points: low_rank 개의 점을 -T_max/2 ~ T_max/2에서 균등 샘플링
        points = np.linspace(-T_max / 2, T_max / 2, self.low_rank)

        total_bases = (self.low_freq_num + self.high_freq_num) * self.phi_num
        # bases 배열: [total_bases, low_rank]
        bases = torch.empty(total_bases, self.low_rank)
        i = 0
        # 먼저 low_freq 그룹: 저주파 성분
        for freq in low_freq:
            for phi in phi_set:
                # 각 점에 대해 cos(freq*x + phi)
                base = torch.tensor([math.cos(freq * x + phi) for x in points])
                bases[i, :] = base
                i += 1
        # 그 다음 high_freq 그룹: 고주파 성분
        for freq in high_freq:
            for phi in phi_set:
                base = torch.tensor([math.cos(freq * x + phi) for x in points])
                bases[i, :] = base
                i += 1

        # 스케일 적용
        bases = self.alpha * bases

        # basis는 고정되어야 하므로, 학습되지 않도록 설정
        bases = nn.Parameter(bases, requires_grad=False)

        # 만약 select_two 옵션이 True이면, low_freq 그룹의 첫 번째 basis와 고_freq 그룹의 마지막 basis를 선택
        if self.select_two:
            # low_freq 그룹은 처음 phi_num rows, 고_freq 그룹는 마지막 phi_num rows.
            # 보통 가장 낮은 주파수 성분은 low_freq 그룹의 첫 번째 row (phi_set의 첫 번째 값)로 가정하고,
            # 가장 높은 주파수 성분은 high_freq 그룹의 마지막 row (phi_set의 마지막 값)로 가정합니다.
            low_component = bases[0:1, :]               # [1, low_rank]
            high_component = bases[-1:, :]              # [1, low_rank]
            selected_bases = torch.cat(
                [low_component, high_component], dim=0)  # [2, low_rank]
            return selected_bases
        else:
            return bases  # [total_bases, low_rank]

    def forward(self):
        return self.bases


#############################################
# 2. BandResHyperNetwork: (band, resolution) -> coeff per layer
#############################################


class BandResHyperNetwork(nn.Module):
    """
    각 배치의 (band_id, resolution) 정보를 받아서,
    n_layers × (n_fourier_bases * low_rank) 차원의 벡터를 생성한 후,
    [B, n_layers, n_fourier_bases, low_rank]로 reshape합니다.
    """

    def __init__(self, n_band: int = 256, embed_dim: int = 16, hidden_dim: int = 64,
                 n_layers: int = 3, n_fourier_bases: int = 25, low_rank: int = 10):
        super().__init__()
        self.n_layers = n_layers
        self.n_fourier_bases = n_fourier_bases
        self.low_rank = low_rank
        self.mr_dim = n_fourier_bases * low_rank  # 총 차원

        self.band_embed = nn.Embedding(n_band, embed_dim)
        self.res_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embed_dim),
            nn.ReLU()
        )
        # 최종 출력 차원: n_layers * (n_fourier_bases * low_rank)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_layers * self.mr_dim)
        )

    def forward(self, band_ids: torch.Tensor, resolution: torch.Tensor):
        """
        band_ids: [B] (정수)
        resolution: [B, 1] (실수)

        출력: coeff ∈ ℝ^(B, n_layers, n_fourier_bases, low_rank)
        """
        B = band_ids.shape[0]
        band_e = self.band_embed(band_ids)         # [B, embed_dim]
        res_e = self.res_mlp(resolution)            # [B, embed_dim]
        cat_e = torch.cat([band_e, res_e], dim=-1)   # [B, 2*embed_dim]
        out = self.fc(cat_e)                        # [B, n_layers * mr_dim]
        coeff = out.view(B, self.n_layers, self.n_fourier_bases, self.low_rank)
        return coeff

#############################################
# 3. LoRAConvLayer: LoRA 기반 Conv weight 계산
#############################################


class LoRAConvLayer(nn.Module):
    """
    LoRA를 이용해 Conv weight를 구성합니다.
    각 배치에 대해, 입력:
        - fourier_mod: [B, n_fourier_bases, low_rank] coefficient (hypernetwork output)와
          fixed basis를 이용해 계산 후, 
          modulation = (fourier_mod)^T @ fixed_basis, yielding [B, low_rank, low_rank]
    최종 weight for each batch:
         W^(b) = weights_alpha @ modulation^(b) @ weights_beta
         where:
           - weights_alpha: [out_channels, low_rank]
           - weights_beta:  [low_rank, in_channels * kernel_size^2]
    배치별 weight를 평균하여 사용한 후, F.conv2d에 적용합니다.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 n_fourier_bases: int, low_rank: int, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_fourier_bases = n_fourier_bases
        self.low_rank = low_rank  # r
        self.mr_dim = n_fourier_bases * low_rank
        self.stride = stride

        # LoRA 파라미터: 앞 뒤 부분은 낮은 차원 (low_rank)로 설정
        self.weights_alpha = nn.Parameter(
            torch.randn(out_channels, low_rank) * 0.02)
        self.weights_beta = nn.Parameter(torch.randn(
            low_rank, in_channels * kernel_size * kernel_size) * 0.02)

    def forward(self, x: torch.Tensor, coeff: torch.Tensor, fixed_basis: torch.Tensor):
        """
        x: [B, in_channels, H, W]
        coeff: [B, n_fourier_bases, low_rank] 해당 레이어에 대해 hypernetwork로부터 나온 coeff
        fixed_basis: [n_fourier_bases, low_rank] (고정된 Fourier basis)

        Steps:
          1. modulation = (coeff^T) @ fixed_basis  (per batch)
             coeff: [B, n_fourier_bases, low_rank] → transpose to [B, low_rank, n_fourier_bases]
             fixed_basis: [n_fourier_bases, low_rank]
             modulation: [B, low_rank, low_rank]
          2. For each batch, compute: W^(b) = weights_alpha @ modulation^(b) @ weights_beta,
             resulting in shape [out_channels, in_channels * kernel_size^2].
          3. Average over batch and reshape to [out_channels, in_channels, kernel_size, kernel_size].
          4. Apply F.conv2d.
        """
        B = x.shape[0]
        # Step 1: Compute modulation per batch
        # coeff: [B, n_fourier_bases, low_rank] → transpose -> [B, low_rank, n_fourier_bases]
        coeff_t = coeff.transpose(1, 2)  # [B, low_rank, n_fourier_bases]
        # fixed_basis: [n_fourier_bases, low_rank] → Multiply: [B, low_rank, n_fourier_bases] @ [n_fourier_bases, low_rank] = [B, low_rank, low_rank]
        # [B, low_rank, low_rank]
        modulation = torch.matmul(coeff_t, fixed_basis)

        # Step 2: Compute weight per batch: W^(b) = alpha @ modulation^(b) @ beta
        # weights_alpha: [out_channels, low_rank] → expand to [B, out_channels, low_rank]
        alpha_exp = self.weights_alpha.unsqueeze(0).expand(
            B, -1, -1)  # [B, out_channels, low_rank]
        # Compute intermediate: [B, out_channels, low_rank] = (alpha_exp @ modulation)
        # [B, out_channels, low_rank]
        modulated_alpha = torch.bmm(alpha_exp, modulation)
        # weights_beta: [low_rank, in_channels * kernel_size^2]
        # Final weight per batch: [B, out_channels, in_channels * kernel_size^2]
        W_batch = torch.bmm(
            modulated_alpha, self.weights_beta.unsqueeze(0).expand(B, -1, -1))

        # Step 3: Reshape per batch to [B, out_channels, in_channels, kernel_size, kernel_size]
        W_batch = W_batch.view(
            B, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        # 단순화를 위해 배치 평균을 취함 (실제 응용에서는 vmap 등의 기법으로 배치별로 다르게 사용할 수 있음)
        # [out_channels, in_channels, kernel_size, kernel_size]
        W_final = W_batch.mean(dim=0)

        # Step 4: conv2d 적용
        padding = self.kernel_size // 2 if self.stride == 1 else 0
        out = F.conv2d(x, W_final, bias=None,
                       stride=self.stride, padding=padding)
        return out

#############################################
# 4. MultiSpectralEmbedding: 전체 파이프라인
#############################################


class MultiSpectralEmbedding(nn.Module):
    """
    최종 파이프라인:
      - 입력: [B, in_channels, H, W] (멀티스펙트럴 데이터)
      - Hyper-Network를 통해, 각 레이어별 coeff를 생성: [B, n_layers, n_fourier_bases, low_rank]
      - Fixed Fourier Basis: [n_fourier_bases, low_rank]
      - 각 레이어마다 modulation = (coeff^T @ fixed_basis) (→ [B, low_rank, low_rank])
      - 3개의 LoRAConvLayer를 순차적으로 적용:
            layer1, layer2: kernel_size=3, stride=1, padding=1
            layer3: patch embedding layer: kernel_size=16, stride=16, padding=0
      - 최종 출력: [B, out_dim, H_out, W_out] (예: H_out = H//16)
    """

    def __init__(self,
                 in_channels: int,
                 hidden_dims: tuple = (32, 64),
                 out_dim: int = 128,
                 n_layers: int = 3,
                 phi_num=32,
                 n_fourier_bases: int = 32*128,  # M
                 low_rank: int = 32,         # r
                 kernel_sizes: tuple = (3, 3, 16)):
        super().__init__()
        self.n_layers = n_layers
        self.n_fourier_bases = n_fourier_bases
        self.low_rank = low_rank
        self.mr_dim = n_fourier_bases * low_rank  # 예: 250
        assert n_fourier_bases % phi_num == 0, "n_fourier_bases must be divisible by phi_num"
        high_freq_num = int(n_fourier_bases/phi_num) // 2
        low_freq_num = int(n_fourier_bases/phi_num - high_freq_num)

        # Fixed Fourier basis: shape [n_fourier_bases, low_rank]
        self.fixed_basis = FourierBasisGenerator(low_rank=low_rank,
                                                 high_freq_num=high_freq_num,
                                                 low_freq_num=low_freq_num,
                                                 phi_num=phi_num,
                                                 alpha=0.05,
                                                 select_two=False)

        # Hyper-Network: 출력 shape [B, n_layers, n_fourier_bases, low_rank]
        self.hypernet = BandResHyperNetwork(n_band=256, embed_dim=16, hidden_dim=64,
                                            n_layers=n_layers, n_fourier_bases=n_fourier_bases,
                                            low_rank=low_rank)

        # 3개의 LoRAConvLayer
        # layer1: kernel_size=3, stride=1, out_channels=hidden_dims[0]
        self.layer1 = LoRAConvLayer(in_channels, hidden_dims[0],
                                    kernel_size=kernel_sizes[0],
                                    n_fourier_bases=n_fourier_bases,
                                    low_rank=low_rank,
                                    stride=1)
        # layer2: kernel_size=3, stride=1, out_channels=hidden_dims[1]
        self.layer2 = LoRAConvLayer(hidden_dims[0], hidden_dims[1],
                                    kernel_size=kernel_sizes[1],
                                    n_fourier_bases=n_fourier_bases,
                                    low_rank=low_rank,
                                    stride=1)
        # layer3 (patch embedding): kernel_size=16, stride=16, out_channels=out_dim
        self.layer3 = LoRAConvLayer(hidden_dims[1], out_dim,
                                    kernel_size=kernel_sizes[2],
                                    n_fourier_bases=n_fourier_bases,
                                    low_rank=low_rank,
                                    stride=kernel_sizes[2])

    def forward(self, x: torch.Tensor, band_ids: torch.Tensor, gsd_vals: torch.Tensor):
        """
        x: [B, in_channels, H, W]
        band_ids: [B] (각 배치 대표 band id; 단순화를 위해 각 배치당 1개)
        gsd_vals: [B, 1] (각 배치 대표 GSD)
        """
        B = x.shape[0]
        # Hyper-network: coeff의 shape: [B, n_layers, n_fourier_bases, low_rank]
        coeff_all = self.hypernet(band_ids, gsd_vals)

        # 고정 Fourier basis: shape: [n_fourier_bases, low_rank]
        fixed_basis = self.fixed_basis()  # [M, r]

        # For each layer l, compute modulation matrix = (coeff[b,l]^T) @ fixed_basis
        # 결과: modulation of shape [B, n_layers, low_rank, low_rank]
        # coeff[b,l]: [n_fourier_bases, low_rank] → transpose: [low_rank, n_fourier_bases]
        modulation = torch.zeros(
            B, self.n_layers, self.low_rank, self.low_rank, device=x.device)
        for l in range(self.n_layers):
            # coeff for layer l: shape [B, n_fourier_bases, low_rank]
            coeff_l = coeff_all[:, l, :, :]  # [B, n_fourier_bases, low_rank]
            # Compute modulation per batch:
            # modulation[b] = (coeff_l[b].transpose(0,1)) @ fixed_basis, shape: [low_rank, low_rank]
            modulation[:, l, :, :] = torch.bmm(coeff_l.transpose(
                1, 2), fixed_basis.unsqueeze(0).expand(B, -1, -1))
            # Alternatively, one can vectorize using torch.einsum:
            # modulation[:, l, :, :] = torch.einsum('bri,ir->brr', coeff_l.transpose(1,2), fixed_basis)

        # Now, 각 레이어마다, 배치별로 modulation 값을 전달
        # (여기서는 각 layer에 대해 배치 모듈레이션을 평균한 후 전달하는 단순화)

        mod1 = modulation[:, 0, :, :].mean(dim=0)  # [low_rank, low_rank]
        out1 = self.layer1(x,
                           coeff=coeff_all[:, 0, :, :].view(
                               B, self.n_fourier_bases, self.low_rank),
                           fixed_basis=fixed_basis)
        # 위 layer1 내부에서 modulation은 재계산하므로, 직접 modulation 사용 대신,
        # 우리는 layer 내부에서 hypernetwork coeff와 fixed_basis를 받아 modulation을 계산합니다.
        # (이미 LoRAConvLayer 구현에서 해당 계산을 수행)
        # 따라서, 여기서는 그냥 전달하면 됩니다.

        out1 = F.relu(out1)
        out2 = self.layer2(out1,
                           coeff=coeff_all[:, 1, :, :].view(
                               B, self.n_fourier_bases, self.low_rank),
                           fixed_basis=fixed_basis)
        out2 = F.relu(out2)
        out3 = self.layer3(out2,
                           coeff=coeff_all[:, 2, :, :].view(
                               B, self.n_fourier_bases, self.low_rank),
                           fixed_basis=fixed_basis)
        return out3


#############################################
# 예시 실행
#############################################
if __name__ == "__main__":
    # 예시: B=2, in_channels=6, H=W=224
    B, in_channels, H, W = 2, 6, 224, 224
    x = torch.randn(B, in_channels, H, W)
    # 각 배치마다 대표 band id와 GSD 값
    band_ids = torch.tensor([8, 15])       # [B]
    gsd_vals = torch.tensor([[10.0], [20.0]])  # [B, 1]

    model = MultiSpectralEmbedding(
        in_channels=in_channels,
        hidden_dims=(32, 64),
        out_dim=768,
        n_layers=3,
        phi_num=8,
        n_fourier_bases=8*128,  # M = 25
        low_rank=16,         # r = 10, so mr_dim = 25*10 = 250
        kernel_sizes=(3, 3, 8)
    )

    out = model(x, band_ids, gsd_vals)
    print("Input shape:", x.shape)      # [2, 6, 224, 224]
    # 최종 layer3은 patch embedding: kernel=16, stride=16 → H_out = 224/16 = 14
    print("Output shape:", out.shape)     # 예상: [2, 128, 14, 14]
