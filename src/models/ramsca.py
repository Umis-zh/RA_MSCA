import torch
import torch.nn.functional as F
from models.msca import MSCA

class RAMSCA(MSCA):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        eps = 0.0001
        temperature = config['reliability_temperature'] or 0.2
        shrinkage = config['reliability_shrinkage'] or 0.5
        image_rel = self._behavioral_consistency(self.image_embedding.weight)
        text_rel = self._behavioral_consistency(self.text_embedding.weight)
        raw_w = 2 * torch.softmax(torch.stack([image_rel, text_rel], dim=1) / temperature, dim=1)
        image_item_w = 1 + shrinkage * (raw_w[:, 0] - 1)
        text_item_w = 1 + shrinkage * (raw_w[:, 1] - 1)
        r_t = self.norm_R
        image_user_w = torch.sparse.mm(r_t, image_item_w[:, None]).squeeze(1)
        text_user_w = torch.sparse.mm(r_t, text_item_w[:, None]).squeeze(1)
        user_denom = image_user_w + text_user_w + 2 * eps
        image_user_w = 2 * (image_user_w + eps) / user_denom
        text_user_w = 2 * (text_user_w + eps) / user_denom
        self.register_buffer('image_node_w', torch.cat([image_user_w, image_item_w]))
        self.register_buffer('text_node_w', torch.cat([text_user_w, text_item_w]))

    def _behavioral_consistency(self, features):
        features = F.normalize(features.detach(), p=2, dim=1)
        user_context = torch.sparse.mm(self.norm_R, features)
        reconstructed = torch.sparse.mm(self.norm_R.transpose(0, 1), user_context)
        reconstructed = F.normalize(reconstructed, p=2, dim=1)
        return torch.sum(features * reconstructed, dim=1)

    def forward(self, test=True):
        image_item_embeds = self.item_id_embedding.weight * self.gate_v(self.image_trs(self.image_embedding.weight))
        text_item_embeds = self.item_id_embedding.weight * self.gate_t(self.text_trs(self.text_embedding.weight))
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = []
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        collab_embeds = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)
        struct_embeds = self.semantic_encode(self.struct_original_adj, self.item_id_embedding.weight)
        image_embeds = self.semantic_encode(self.image_original_adj, image_item_embeds)
        text_embeds = self.semantic_encode(self.text_original_adj, text_item_embeds)
        image_weighted = image_embeds * self.image_node_w[:, None]
        text_weighted = text_embeds * self.text_node_w[:, None]
        multi_embed_list = [image_weighted, text_weighted, struct_embeds]
        attn_weights = self.softmax(torch.cat([self.query_common(embed) for embed in multi_embed_list], dim=-1))
        redundant_embeds = sum((w.unsqueeze(1) * embed for w, embed in zip(attn_weights.unbind(dim=1), multi_embed_list)))
        multi_embeds = sum(multi_embed_list) - redundant_embeds
        final_embeds = collab_embeds + self.fusion_coeff * multi_embeds
        final_user_embeds, final_item_embeds = torch.split(final_embeds, [self.n_users, self.n_items], dim=0)
        if test:
            return (final_user_embeds, final_item_embeds)
        return (final_user_embeds, final_item_embeds, collab_embeds, struct_embeds, image_embeds, text_embeds)

    def cal_cl_loss(self, embed_1, embed_2, tau, weights=None):
        embed_1 = F.normalize(embed_1, p=2, dim=1)
        embed_2 = F.normalize(embed_2, p=2, dim=1)
        pos = torch.sum(embed_1 * embed_2, dim=-1)
        tot = torch.matmul(embed_1, embed_2.transpose(0, 1))
        per_row = torch.logsumexp((tot - pos[:, None]) / tau, dim=1)
        if weights is None:
            return per_row.mean()
        weights = weights.detach().clamp_min(0.0001)
        return (per_row * weights).sum() / weights.sum()

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = (interaction[0], interaction[1], interaction[2])
        final_u, final_i, collab, struct, image, text = self.forward(test=False)
        bpr_loss = self.cal_bpr_loss(final_u[users], final_i[pos_items], final_i[neg_items])
        reg_loss = self.cal_reg_loss()
        collab_u, collab_i = torch.split(collab, [self.n_users, self.n_items], dim=0)
        struct_u, struct_i = torch.split(struct, [self.n_users, self.n_items], dim=0)
        image_u, image_i = torch.split(image, [self.n_users, self.n_items], dim=0)
        text_u, text_i = torch.split(text, [self.n_users, self.n_items], dim=0)
        image_u_w, image_i_w = torch.split(self.image_node_w, [self.n_users, self.n_items], dim=0)
        text_u_w, text_i_w = torch.split(self.text_node_w, [self.n_users, self.n_items], dim=0)
        m_u = self.cal_cl_loss(collab_u[users], image_u[users], self.tau, image_u_w[users])
        m_u += self.cal_cl_loss(collab_u[users], text_u[users], self.tau, text_u_w[users])
        m_i = self.cal_cl_loss(collab_i[pos_items], image_i[pos_items], self.tau, image_i_w[pos_items])
        m_i += self.cal_cl_loss(collab_i[pos_items], text_i[pos_items], self.tau, text_i_w[pos_items])
        c_u = self.cal_cl_loss(collab_u[users], struct_u[users], self.tau)
        c_i = self.cal_cl_loss(collab_i[pos_items], struct_i[pos_items], self.tau)
        return bpr_loss + self.cl_weight * (c_u + c_i + m_u + m_i) + self.reg_weight * reg_loss
