import torch
from torch import nn
from torch.nn import functional as F


class DSMIL_Attention(nn.Module):
    def __init__(self):
        super().__init__()

    # q(patch_num, size[2]), q_max(num_classes, size[2])
    def forward(self, q, q_max):
        attn = q @ q_max.transpose(1, 0) # (patch_num, num_classes)

        return F.softmax(attn / torch.sqrt(torch.tensor(q.shape[1], dtype=torch.float32)), dim=0) # (patch_num, num_classes)


class DSMIL_BClassifier(nn.Module):
    def __init__(self,
        num_classes: int,
        size = [768, 128, 128],
        dropout: float = 0.5,
    ):
        super().__init__()

        self.query_weight = nn.Sequential(
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Linear(size[1], size[2]),
            nn.Tanh()
        )
        self.value_weight = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(size[0], size[0]),
            nn.ReLU()
        )
        self.attention = DSMIL_Attention()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(num_classes, num_classes, kernel_size=size[0])
        )

    # features(patch_num, feature_dim), instance_logits(patch_num, num_classes)
    ## feature_dim == size[0]
    def forward(self, features, instance_logits):
        value = self.value_weight(features)  # (patch_num, size[0])
        query = self.query_weight(features)  # (patch_num, size[2])

        # sort class scores along the instance dimension, idxs : (patch_num, num_classes)
        _, idxs = torch.sort(instance_logits, dim=0, descending=True)

        # select critical features, feature_max : (num_classes, size[0])
        feature_max = features[idxs[0]]

        # compute queries of critical instances, query_max : (num_classes, size[2])
        query_max = self.query_weight(feature_max)

        attn_score = self.attention(query, query_max)  # (patch_num, num_classes)

        bag_feature = attn_score.transpose(1, 0) @ value # compute bag representation, (num_classes, size[0])
        bag_logits = self.classifier(bag_feature)[:, 0] # (num_classes,)

        return bag_logits, attn_score, bag_feature

        """

        (Pdb++) idxs
        tensor([[1798, 2078, 2594,  892, 2286],
                [1567, 1925, 1984,  390, 2452],
                [1720, 2077, 2293,   46, 2214],
                ...,
                [2588, 1509,  515,  549,  812],
                [1786, 1583,   74, 1325, 2710],
                [2589, 1679,  468, 2081, 2663]], device='cuda:0')

        (Pdb++) max(instance_logits[:, 0])
        tensor(0.1039, device='cuda:0', dtype=torch.float16, grad_fn=<UnbindBackward0>)
        (Pdb++) max(instance_logits[:, 1])
        tensor(0.1204, device='cuda:0', dtype=torch.float16, grad_fn=<UnbindBackward0>)
        (Pdb++) max(instance_logits[:, 2])
        tensor(0.0464, device='cuda:0', dtype=torch.float16, grad_fn=<UnbindBackward0>)
        (Pdb++) max(instance_logits[:, 3])
        tensor(0.2222, device='cuda:0', dtype=torch.float16, grad_fn=<UnbindBackward0>)
        (Pdb++) max(instance_logits[:, 4])
        tensor(0.2566, device='cuda:0', dtype=torch.float16, grad_fn=<UnbindBackward0>)

        (Pdb++) instance_logits[1798]
        tensor([ 0.1039, -0.0742, -0.0269,  0.1268,  0.0764], device='cuda:0',
            dtype=torch.float16, grad_fn=<SelectBackward0>)
        (Pdb++) instance_logits[2078]
        tensor([-0.0253,  0.1204, -0.0557,  0.0348,  0.1653], device='cuda:0',
            dtype=torch.float16, grad_fn=<SelectBackward0>)
        (Pdb++) instance_logits[2594]
        tensor([-0.0557, -0.0510,  0.0464,  0.0891,  0.1752], device='cuda:0',
            dtype=torch.float16, grad_fn=<SelectBackward0>)
        (Pdb++) instance_logits[892]
        tensor([-0.0338, -0.0707, -0.0448,  0.2222,  0.0554], device='cuda:0',
            dtype=torch.float16, grad_fn=<SelectBackward0>)
        (Pdb++) instance_logits[2286]
        tensor([-0.0319,  0.0011, -0.0218,  0.1144,  0.2566], device='cuda:0',
            dtype=torch.float16, grad_fn=<SelectBackward0>)

        """


class DSMIL(nn.Module):
    def __init__(self,
        num_classes: int,
        size = [768, 128, 128],
        dropout: float = 0.5,
    ):
        super().__init__()

        self.i_classifier = nn.Linear(size[0], num_classes)
        self.b_classifier = DSMIL_BClassifier(num_classes, size, dropout)

    def forward(self, features):
        # features : (patch_num, feature_dim)

        inst_logits = self.i_classifier(features) # (patch_num, num_classes)
        bag_logits, attn, bag_feature = self.b_classifier(features, inst_logits)

        # (num_classes,), (N, num_classes),
        return bag_logits, inst_logits, attn, bag_feature

