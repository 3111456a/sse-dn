import datetime
import os
import joblib  # 新增：用于保存矩阵到本地
import numpy as np
import pkbar
import torch
import torch_geometric
from local_tsl.nn.blocks.encoders.recurrent.agcrn import AGCRN

from sse_denoiser.spatio_temporal_transformer import Transformer as SpatioTemporalTransformer
from utils import denoising_plots
from geopy.distance import geodesic # 新增，用于计算地球表面真实弧面距离

class STAGRNNDenoiserModel(torch.nn.Module):
    """
    STAGRNNDenoiserModel is a denoising model using Spatio-Temporal Attention and Graph Recurrent Neural Networks.
    (Refactored: Removed legacy residual and static learning branches)
    """

    def __init__(self, return_attention=False, add_transformer=True,
                 use_spatial_attention=True, use_temporal_attention=True):
        super(STAGRNNDenoiserModel, self).__init__()
        self.n_components = 2
        self.transformer_dropout = 0.1
        self.adj_embedding_dim = 32
        self.add_transformer = add_transformer
        self.rnn_hid_size = 128
        self.n_nodes = 200
        self.return_attention = return_attention

        self.use_spatial_attention = use_spatial_attention
        self.use_temporal_attention = use_temporal_attention

        rnn_input = self.n_components
        
        # AGCRN: Data-driven spatial feature extraction
        self.agcrn = AGCRN(input_size=rnn_input, emb_size=self.adj_embedding_dim,
                           hidden_size=self.rnn_hid_size, num_nodes=self.n_nodes)

        # Spatio-Temporal Transformer Layer
        if self.add_transformer:
            transf_axis = 'both'
            if not self.use_temporal_attention and self.use_spatial_attention:
                transf_axis = 'nodes'
            elif self.use_temporal_attention and not self.use_spatial_attention:
                transf_axis = 'time'

            self.transformer = SpatioTemporalTransformer(
                input_size=self.rnn_hid_size, hidden_size=self.rnn_hid_size,
                axis=transf_axis, causal=False,
                dropout=self.transformer_dropout,
                return_attention=return_attention
            )

        self.lin = torch.nn.Linear(self.rnn_hid_size, self.n_components)

    def forward(self, x_in, visualize_rnn_feat=False, visualize_transf_feat=False):
        # permute (B, N, time, feat) ---> (B, time, N, feat)
        x_perm = torch.permute(x_in, (0, 2, 1, 3))

        x_out, h = self.agcrn(x_perm)
        if visualize_rnn_feat:
            return x_out

        if self.add_transformer:
            if self.return_attention:
                x_out, statt = self.transformer(x_out)
                t_att, s_att = statt
                return x_out, (t_att, s_att)
            else:
                x_out = self.transformer(x_out)

        # permute again from (B, time, N, feat) ---> (B, N, time, feat)
        x_out = torch.permute(x_out, (0, 2, 1, 3))
        if visualize_transf_feat:
            return x_out

        x_out = torch.nn.functional.dropout(x_out, p=0.5, training=self.training)
        x_out = self.lin(x_out)

        return x_out


class STAGRNNDenoiser:
    """
    STAGRNNDenoiser is a wrapper class for training and using the STAGRNNDenoiserModel (SSEdenoiser).
    """

    def __init__(self, **kwargs):
        self.model = None
        self.n_stations = kwargs['n_stations']
        self.window_length = kwargs['window_length']
        self.n_directions = kwargs['n_directions']
        self.n_epochs = kwargs.get('n_epochs', 500)
        self.batch_size = kwargs.get('batch_size', 16)
        self.patience = kwargs.get('patience', 50)
        self.initial_learning_rate = kwargs.pop('learning_rate', None)

        self.train_verbosity_level = kwargs.get('verbosity', 0)
        self.station_coordinates = kwargs.get('station_coordinates', None)
        self.custom_loss = kwargs.get('custom_loss', False)
        self.val_catalogue = kwargs.get('val_catalogue', None)
        self.custom_loss_coeff = kwargs.get('custom_loss_coeff', 1)

<<<<<<< HEAD
        # ---------- 新增：空间损失开关与系数 ----------
        self.use_spatial_loss = kwargs.get('use_spatial_loss', True)
        # 惩罚系数非常重要，用于平衡 MSE 和 空间NCC 的量级
        self.spatial_loss_coeff = kwargs.get('spatial_loss_coeff', 0.01) 
        # ---------------------------------------------
=======

        # ---------- 新增：软阈值空间惩罚超参数 ----------
        self.d_threshold = kwargs.get('d_threshold', 400.0)
        self.alpha = kwargs.get('alpha', 0.05)
        # 默认保存到工作目录下的 weights 文件夹中
        self.spatial_weights_path = kwargs.get('spatial_weights_path', './spatial_weights.joblib')
        self.W_matrix = None
        # -----------------------------------------------

>>>>>>> 15f65024881de2b7efc7ff3f786149936e5b9413

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.weight_path = None
        self.log_dir = None
        self.tb_writer = None
        self.img_writer = None
        self.device = None

        self.y_val = kwargs.pop('y_val', None)
        self.y_test = kwargs.pop('y_test', None)
        self.scaler = kwargs.pop('scaler', None)

        # Network architecture kwargs
        self.return_attention = kwargs.pop('return_attention', False)
        self.add_transformer = kwargs.pop('add_transformer', True)
        self.use_temporal_attention = kwargs.pop('use_temporal_attention', True)
        self.use_spatial_attention = kwargs.pop('use_spatial_attention', True)
        self.graph_loader = kwargs.get('graph_loader', False)
        self.spatial_weights_path = r'/root/autodl-tmp/data/spatial_weights.joblib'

    def _compute_spatial_weights(self):
        """
        计算并保存静态空间权重矩阵 W_ij
        """
        if self.station_coordinates is None:
            raise ValueError("错误：未提供 station_coordinates，无法计算空间权重。")

        # 1. 获取台站总数 N (例如 200)
        coords = self.station_coordinates
        N = coords.shape[0]
        
        # 2. 计算 D_{i,j} 距离矩阵 (单位: km)
        # 注意：geodesic 接收 (lat, lon)
        dist_matrix = np.zeros((N, N))
        print(f"正在为 {N} 个台站预计算球面距离矩阵...")
        
        for i in range(N):
            for j in range(i + 1, N): # 利用对称性减少一半计算量
                d = geodesic(
                    (coords[i, 0], coords[i, 1]), 
                    (coords[j, 0], coords[j, 1])
                ).km
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # 3. 计算 W_{i,j} 平滑惩罚权重
        # 公式: W = 1 / (1 + exp(-alpha * (D - D_threshold)))
        W = 1.0 / (1.0 + np.exp(-self.alpha * (dist_matrix - self.d_threshold)))
        
        # 4. 物理约束：自己对自己不计算相关性惩罚 (对角线设为0)
        np.fill_diagonal(W, 0.0)

        # 5. 保存到本地
        save_dir = os.path.dirname(self.spatial_weights_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        joblib.dump({
            'weight_matrix': W,
            'distance_matrix': dist_matrix,
            'alpha': self.alpha,
            'd_threshold': self.d_threshold,
            'station_coordinates': coords
        }, self.spatial_weights_path)
        
        print(f"成功：空间权重矩阵已保存至 {self.spatial_weights_path}")
        return W

    def setup_spatial_weights(self):
        """
        统筹空间权重的获取：
        1. 检查本地是否存在保存好的权重文件。
        2. 若存在，直接读取；若不存在，调用计算函数并保存。
        3. 将 numpy 矩阵转换为 tensor 并挂载到 GPU/CPU。
        """
        if os.path.exists(self.spatial_weights_path):
            print(f"-> 检测到本地缓存，直接读取空间权重矩阵: {self.spatial_weights_path}")
            data = joblib.load(self.spatial_weights_path)
            w_np = data['weight_matrix']
        else:
            print("-> 未检测到本地缓存，开始首次计算空间权重矩阵...")
            w_np = self._compute_spatial_weights()  # 这个函数里已经包含了 joblib.dump 保存逻辑
            
        # 将矩阵转换为 PyTorch Tensor，并发送到模型所在的 Device
        self.W_matrix = torch.tensor(w_np, dtype=torch.float32, device=self.device, requires_grad=False)
        print("-> 静态空间距离权重矩阵 (W_matrix) 已成功挂载至设备。")

    def _callbacks(self, stagename):
        from torch.utils.tensorboard import SummaryWriter
        base_checkpoint_path = os.path.expandvars('$WORK') + '/models'
        base_weight_dir = os.path.expandvars('$WORK') + '/weights'

        checkpoint_path = os.path.join(base_checkpoint_path, 'SSEdetector_char')
        weight_dir = os.path.join(base_weight_dir, 'SSEdetector_char')

        for path in [base_checkpoint_path, base_weight_dir, checkpoint_path, weight_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

        date_string = datetime.datetime.now().strftime("%d%b%Y-%H%M%S")
        log_dir = os.path.expandvars('$WORK') + "/logs/fit/cascadia" + date_string + f'_{stagename}'

        self.weight_path = os.path.join(weight_dir, f'best_cascadia_{date_string}_{stagename}.pt')
        self.tb_writer = SummaryWriter(log_dir)
        self.img_writer = SummaryWriter(os.path.join(log_dir, 'img'))

    def load_spatial_weights(self):
        import joblib
        if os.path.exists(self.spatial_weights_path):
            print(f"Loading spatial weights from {self.spatial_weights_path}")
            data = joblib.load(self.spatial_weights_path)
            w_np = data['weight_matrix']
            self.W_matrix = torch.tensor(w_np, dtype=torch.float32, device=self.device, requires_grad=False)
        else:
            raise FileNotFoundError(f"未找到空间权重文件: {self.spatial_weights_path}。请先运行生成脚本。")
    def build(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = STAGRNNDenoiserModel(return_attention=self.return_attention,
                                     add_transformer=self.add_transformer,
                                     use_spatial_attention=self.use_spatial_attention,
                                     use_temporal_attention=self.use_temporal_attention)
        model.to(device)
        self.device = device
        self.model = model
<<<<<<< HEAD
        # --- 必须加这一行，把计算好的矩阵丢进 GPU ---
        if self.use_spatial_loss:
            self.load_spatial_weights()
=======
        # ---------- 新增：初始化空间权重矩阵 (一行调用，方便注释) ----------
        if self.custom_loss:
            self.setup_spatial_weights() 
        # ----------------------------------------------------------------
>>>>>>> 15f65024881de2b7efc7ff3f786149936e5b9413

    def summary(self, x):
        print(torch_geometric.nn.summary(self.model, x.to(self.device), max_depth=1))
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        pytorch_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total params:', pytorch_total_params)
        print('Trainable params:', pytorch_trainable_params)
        print('Non-trainable params:', pytorch_total_params - pytorch_trainable_params)
        print('-' * 20)

    def summary_nograph(self, x):
        """兼容 train.py 中显式调用的无图版本 summary"""
        self.summary(x)

    def minibatch_train_nograph(self):
        """防止 train.py 显式调用无图版训练"""
        self.minibatch_train()

    def _inference_nograph(self):
        """防止 train.py 显式调用无图版推理"""
        return self.inference()

    def ang_loss(self, ts_true, ts_pred, disp_true, disp_pred):
        """MSE for denoised time series and angular similarity for static displacement fields."""
        ts_misfit = torch.nn.functional.mse_loss(ts_true, ts_pred)

        cosine = torch.nn.functional.cosine_similarity(disp_true, disp_pred, dim=-1)
        clamped_cosine = torch.clamp(cosine, -1.0 + 1e-07, 1.0 - 1e-07)
        arccos = torch.arccos(clamped_cosine)
        angular_misfit = arccos.mean()

        return ts_misfit + self.custom_loss_coeff * angular_misfit

    # 新添加的带空间物理约束的loss，距离越大的台站如果显示出近似的位移速率，loss上升，为了区分共模噪声和sse
    def spatio_temporal_loss(self, ts_pred,ts_true):
        """
        综合损失函数: MSE + 软阈值空间去相关惩罚 (Soft-Margin Distance Decay Loss)
        """
        # 1. 基础重建损失
        mse_loss = torch.nn.functional.mse_loss(ts_pred, ts_true)

        spatial_penalty = 0.0
        
        if self.use_spatial_loss:
            if self.W_matrix is None:
                raise ValueError("未找到空间权重矩阵 self.W_matrix")
                
            # 【第二类：动态预测参数】计算近似速度
            # 维度: (Batch, N, Time_diff, Feat)
            v_pred = ts_pred[:, :, 1:, :] - ts_pred[:, :, :-1, :]
            
            # 提取时间轴长度 (也就是公式隐含的 T)
            T_diff = v_pred.shape[2] 
            
            # 【第三类：统计量参数】
            mu_v = torch.mean(v_pred, dim=2, keepdim=True)
            v_centered = v_pred - mu_v
            sigma_v = torch.std(v_pred, dim=2, unbiased=False, keepdim=True)
            epsilon = 1e-8
            
            # 归一化序列: (Batch, N, Time_diff, Feat)
            v_norm = v_centered / (sigma_v + epsilon)
            
            # -----------------------------------------------------
            # 【完美对齐公式的矩阵运算】
            # -----------------------------------------------------
            # 1. 计算 NCC (归一化互相关) 矩阵
            # 用 einsum 计算批次内所有台站对在时间轴 (t) 上的点积，保留 (b:Batch, i:站i, j:站j, f:特征)
            # 除以 T_diff 确保结果严格落在 [-1, 1] 之间！
            ncc_matrix = torch.einsum('bitf,bjtf->bijf', v_norm, v_norm) / T_diff
            
            # 2. 纯形态平方 (括号外的 ^2)
            # 将相关系数映射到 [0, 1]，无论是正相关还是负相关，都会产生惩罚
            ncc_squared = ncc_matrix ** 2
            
            # 3. 施加空间惩罚权重 (W_i,j)
            # self.W_matrix 维度是 (N, N)，需要 reshape 匹配 (Batch, N, N, Feat) 以便广播相乘
            W_expanded = self.W_matrix.unsqueeze(0).unsqueeze(-1)
            penalty_matrix = W_expanded * ncc_squared
            
            # 4. 外层求和与平均 (1 / N(N-1) \sum_{i!=j})
            # - 在维度 1 (i) 和 2 (j) 上求和，即 sum_{i!=j} 
            #   (因为在预处理时，W_matrix 对角线已经是 0 了，所以可以直接整体 sum)
            # - 除以 N * (N - 1)
            N = self.W_matrix.shape[0]
            spatial_penalty_per_batch = torch.sum(penalty_matrix, dim=(1, 2)) / (N * (N - 1))
            
            # 最后对 Batch 维度和 Feature (东西向/南北向) 维度求均值，化为一个标量
            spatial_penalty = torch.mean(spatial_penalty_per_batch)

        # 最终整合 (spatial_loss_coeff 为超参数，用于平衡两个 Loss)
        total_loss = mse_loss + self.spatial_loss_coeff * spatial_penalty
        # --- 新增：把当前 batch 的分项 loss 存入实例属性 ---
        self.current_mse = mse_loss.item()
        self.current_spatial = spatial_penalty.item() if isinstance(spatial_penalty, torch.Tensor) else spatial_penalty
        
        return total_loss



    def associate_optimizer(self):
        """绑定优化器与损失函数"""
        if self.use_spatial_loss:
            print(f"-> 启用空间去相关约束损失 (系数: {self.spatial_loss_coeff})")
            self.loss = self.spatio_temporal_loss
        elif self.custom_loss:
            self.loss = self.ang_loss
        else:
            self.loss = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_learning_rate)

    def set_callbacks(self, train_codename):
        self._callbacks(train_codename)

    def set_data_loaders(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_epoch(self, progress_bar):
        running_loss = 0.
        running_mse = 0.      # 新增
        running_spatial = 0.  # 新增

        for i, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            x = data[0].to(self.device)
            outputs = self.model(x)

            if not self.custom_loss:
                y = data[1].to(self.device)
                loss = self.loss(outputs, y) # 注意：保持你修复过的参数顺序
            else:
                y = data[1].to(self.device)
                y_disp = (data[1][:, :, -1, :] - data[1][:, :, 0, :]).to(self.device)
                pred_disp = (outputs[:, :, -1, :] - outputs[:, :, 0, :]).to(self.device)
                loss = self.loss(y, outputs, y_disp, pred_disp)

            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()

            # --- 新增：读取并累加当前 batch 的分项 Loss ---
            curr_mse = getattr(self, 'current_mse', 0.0)
            curr_spat = getattr(self, 'current_spatial', 0.0)
            running_mse += curr_mse
            running_spatial += curr_spat

            # --- 修改：让进度条实时显示这三个指标 ---
            progress_bar.update(i, values=[
                ("loss", loss.item()), 
                ("mse", curr_mse), 
                ("spatial", curr_spat)
            ])

        # --- 修改：在 Epoch 结束时，把三个平均值一起返回 ---
        n_batches = len(self.train_loader)
        return running_loss / n_batches, running_mse / n_batches, running_spatial / n_batches
# 实现了早停
    def minibatch_train(self):
        best_vloss = 1_000_000.
        
        # --- 新增：初始化早停计数器 ---
        epochs_no_improve = 0 
        
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}")
            progress_bar = pkbar.Kbar(target=len(self.train_loader), always_stateful=False, width=25,
                                      verbose=self.train_verbosity_level)

            self.model.train()
            avg_train_loss, avg_train_mse, avg_train_spatial = self.train_epoch(progress_bar)
            progress_bar.add(1)
            
            self.model.eval()
            with torch.no_grad():
                running_vloss = 0.0
                end_of_epoch_val_pred = np.zeros(self.y_val.shape)
                n_val_batches = len(self.val_loader)

                for i, vdata in enumerate(self.val_loader):
                    voutputs = self.model(vdata[0].to(self.device))

                    if not self.custom_loss:
                        # 注意：验证集也需要修正参数顺序 (ts_pred, ts_true)
                        vloss = self.loss(voutputs, vdata[1].to(self.device)).item()
                    else:
                        y = vdata[1].to(self.device)
                        y_disp = (vdata[1][:, :, -1, :] - vdata[1][:, :, 0, :]).to(self.device)
                        pred_disp = (voutputs[:, :, -1, :] - voutputs[:, :, 0, :]).to(self.device)
                        vloss = self.loss(voutputs, y, pred_disp, y_disp).item()

                    running_vloss += vloss

                    if i < n_val_batches - 1:
                        end_of_epoch_val_pred[
                            i * self.batch_size:(i + 1) * self.batch_size] = voutputs.cpu().detach().numpy()
                    else:
                        end_of_epoch_val_pred[i * self.batch_size:] = voutputs.cpu().detach().numpy()

                avg_vloss = running_vloss / (i + 1)

            # Tensorboard Logs
            self.tb_writer.add_scalars('', {'train': avg_train_loss, 'validation': avg_vloss}, epoch + 1)
            self.tb_writer.flush()

            figure_disp = denoising_plots(self.val_loader, self.y_val, end_of_epoch_val_pred,
                                          self.val_catalogue, self.station_coordinates, static=False)
            self.img_writer.add_figure('denoising', figure_disp, epoch + 1, close=True)
            self.img_writer.flush()

            progress_bar.add(1, values=[
                ("loss", avg_train_loss),
                ("val_loss", avg_vloss),
                ("mse", avg_train_mse),
                ("spatial", avg_train_spatial)
            ])

            # --- 修改：早停判断与权重保存逻辑 ---
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                epochs_no_improve = 0  # 重置计数器
                torch.save(self.model.state_dict(), self.weight_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"\nEarly stopping triggered! Validation loss hasn't improved for {self.patience} epochs.")
                    break # 跳出 epoch 循环

    def inference(self):
        test_pred = np.zeros(self.y_test.shape)
        if self.custom_loss:
            disp_f_pred = np.zeros((self.y_test.shape[0], self.n_stations, self.n_directions))

        n_test_batches = len(self.test_loader)
        self.model.eval()

        with torch.no_grad():
            for i, tdata in enumerate(self.test_loader):
                print(f'Batch {i + 1}/{n_test_batches}')
                toutputs = self.model(tdata[0].to(self.device))

                if self.custom_loss:
                    dfoutput = toutputs[:, :, -1, :] - toutputs[:, :, 0, :]

                if i < n_test_batches - 1:
                    test_pred[i * self.batch_size:(i + 1) * self.batch_size] = toutputs.cpu().detach().numpy()
                    if self.custom_loss:
                        disp_f_pred[i * self.batch_size:(i + 1) * self.batch_size] = dfoutput.cpu().detach().numpy()
                else:
                    test_pred[i * self.batch_size:] = toutputs.cpu().detach().numpy()
                    if self.custom_loss:
                        disp_f_pred[i * self.batch_size:] = dfoutput.cpu().detach().numpy()

        if not self.custom_loss:
            return test_pred
        else:
            return test_pred, disp_f_pred

    def train(self):
        try:
            self.minibatch_train()
        except KeyboardInterrupt:
            print("Training of SSEdenoiser interrupted and completed")

    def load_weights(self, weight_path, strict=True):
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device), strict=strict)

    def get_model(self):
        return self.model