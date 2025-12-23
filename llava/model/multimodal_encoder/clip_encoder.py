import torch
import torch.nn as nn
from ReConV2.models.ReCon import ReCon2
from ReConV2.utils.config import cfg_from_yaml_file


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.cfg_path = vision_tower
        self.vision_tower_path = args.vision_tower_path
        self.config = cfg_from_yaml_file(self.cfg_path)
        self.config.with_color = args.with_color
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.vision_tower = ReCon2(self.config.model)
        self.hidden_size = self.vision_tower.embed_dim
        self.global_query_num = self.vision_tower.global_query_num
        self.is_loaded = False

    def load_model(self):
        ckpt = torch.load(self.vision_tower_path, map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.vision_tower.load_state_dict(state_dict, strict=True)
        self.vision_tower.requires_grad_(False)

        # ✅ 전체 모델을 cuda:0으로
        self.vision_tower = self.vision_tower.to("cuda:0")

        # ✅ 핵심: embed 모듈도 수동으로 이동
        self.vision_tower.model.embed = self.vision_tower.model.embed.to("cuda:0")

        print(f"[INFO] vision_tower + embed moved to cuda:0")
        self.is_loaded = True



    @torch.no_grad()
    def forward(self, pts):
        # print("🧠 [DEBUG] clip_encoder.forward() called")
        # print(f" - self.device: {self.device}")
        # print(f" - pts type: {type(pts)}")

        if isinstance(pts, list):
            print(" - pts is a list")
            pos_features = []
            local_features = []
            global_features = []
            for i, pt in enumerate(pts):
                print(f"   - pt[{i}].device (before to()): {pt.device}")
                pt_on_device = pt.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                print(f"   - pt[{i}].device (after to()): {pt_on_device.device}")

                pos_feature, local_feature, global_feature = self.vision_tower.model.inference(pt_on_device)
                print(f"   - pos_feature.device: {pos_feature.device}")
                print(f"   - local_feature.device: {local_feature.device}")
                print(f"   - global_feature.device: {global_feature.device}")

                pos_features.append(pos_feature.to(pts[0].dtype))
                local_features.append(local_feature.to(pts[0].dtype))
                global_features.append(global_feature.to(pts[0].dtype))
        else:
            print(f" - pts.device (before to()): {pts.device}")
            pts_on_device = pts.to(device=self.device, dtype=self.dtype)
            print(f" - pts.device (after to()): {pts_on_device.device}")

            pos_features, local_features, global_features = self.vision_tower.model.inference(pts_on_device)

            print(f" - pos_features.device: {pos_features.device}")
            print(f" - local_features.device: {local_features.device}")
            print(f" - global_features.device: {global_features.device}")

            local_features = local_features.to(pts.dtype)
            global_features = global_features.to(pts.dtype)

        return pos_features, local_features, global_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def num_patches(self):
        return self.config.num_group
