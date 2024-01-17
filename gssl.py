# =====================================================================
# Based on moco-v3/moco/builder.py
# https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py
# =====================================================================

# =====================================================================
# Based on moco-v3/moco/builder.py , and SSiT
# https://github.com/YijinHuang/SSiT
# =====================================================================


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from vits import archs


def build_model(args):
    assert args.arch in archs.keys(), 'Not implemented architecture.'
    encoder = partial(
        archs[args.arch],
        pretrained=args.pretrained,
        img_size=args.input_size,
        mask_ratio=args.mask_ratio,
    )

    model = GSSL(
        encoder,
        dim=256,
        mlp_dim=4096,
        T=args.temperature,
        pool_mode=args.pool_mode,
        saliency_threshold=args.saliency_threshold,
    )

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], gradient_as_bucket_view=True)
    else:
        model = model.to(args.device)

    return model


class GSSL(nn.Module):
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, pool_mode='max', saliency_threshold=0.25):
        super(SSiT, self).__init__()

        self.T = T
        self.saliency_threshold = saliency_threshold

        # build encoders
        # base_encoder: Query encoder
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        
        # momentum_encoder = key encoder
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        patch_size = self.base_encoder.patch_size
        if pool_mode == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
        elif pool_mode == 'max':
            self.pool = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)
        else:
            self.pool = None

        self.build_saliency_segmentor_mlps(mlp_dim, patch_size)
        self.build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))

                # # Residual connection
                if dim1 == dim2:
                    # Add a skip connection only if the dimensions match
                    residual_connection = nn.Identity()
                    mlp.append(residual_connection)
                    
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors: 2 projectores

        # Query encoder: 
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # key encoder: momentum_based moving average of the learnable query encoder
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor at Query encoder only
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

    def build_saliency_segmentor_mlps(self, mlp_dim, patch_size):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        self.saliency_segmentor = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=patch_size ** 2,
                kernel_size=1,
            ),
            nn.PixelShuffle(upscale_factor=patch_size),
        )

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        k = concat_all_gather(k)

        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        labels = (torch.arange(N, dtype=torch.long) + N * rank).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def saliency_segmentation_loss(self, f, m):
        f = f[:, 1:]
        m = (m > self.saliency_threshold).float()

        B, L, C = f.shape
        H = W = int(L ** 0.5)
        f = f.permute(0, 2, 1).reshape(B, C, H, W)
        ss = self.saliency_segmentor(f)

        bce = F.binary_cross_entropy_with_logits(ss, m)
        return bce

    def forward(self, x1, x2, m1, m2, m):
        # Input Parameters:
        # x1, x2: Input data for two views.
        # m1, m2: Saliency masks corresponding to the input images for the two views.
        # m: Saliency mask for updating the momentum encoder.

# Optional Pooling:
# mp1, mp2: These are the pooled versions of the saliency masks using
# a pooling operation specified by self.pool.
# The pooling operation is applied if self.pool is not None.

        mp1 = None if self.pool is None else self.pool(m1)
        mp2 = None if self.pool is None else self.pool(m2)


# Feature Computation:
# t1, f1: Features obtained by passing x1 through a base encoder.
# t2, f2: Features obtained by passing x2 through the same base encoder.

# q1: Predictions obtained by passing t1 through a predictor.
# q2: Predictions obtained by passing t2 through the same 
        
        # compute features
        t1, f1 = self.base_encoder(x1)
        t2, f2 = self.base_encoder(x2)
        q1 = self.predictor(t1)
        q2 = self.predictor(t2)

        with torch.no_grad():  # no gradient
            # Momentum Encoder Update:
            # This updates the momentum encoder using the saliency mask m 
            # The update is performed without gradient computation
            # (torch.no_grad() is used):
            # indicating that this part is not intended for backpropagation.
            
            self._update_momentum_encoder(m)  # update the momentum encoder


# Compute Momentum Features as Targets:
# k1, _: Features obtained by passing x1 through the momentum encoder,
# using the pooled mask mp1.
# k2, _: Features obtained by passing x2 through the momentum encoder,
# using the pooled mask mp2.

            # compute momentum features as targets
            k1, _ = self.momentum_encoder(x1, mp1)
            k2, _ = self.momentum_encoder(x2, mp2)


# Loss Computation:
# cl_loss: Contrastive loss is computed using the predictions (q1, q2)
# and momentum features (k1, k2).
# This loss encourages the model to bring together representations
# of semantically similar samples and push apart representations of dissimilar samples.


        cl_loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        
# sp_loss: Saliency segmentation loss is computed using the features (f1, f2)
# and their corresponding saliency masks (m1, m2).
# This loss encourages the model to generate saliency maps that align
# with ground truth saliency masks.
#         
        sp_loss = self.saliency_segmentation_loss(f1, m1) + self.saliency_segmentation_loss(f2, m2)
        
#  Output:
# Returns the:
# 1- contrastive loss (cl_loss)
# 2- saliency segmentation loss (sp_loss)
        return cl_loss, sp_loss


@torch.no_grad()
def concat_all_gather(tensor):
    if not torch.distributed.is_initialized():
        return tensor

    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
