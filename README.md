## 编译运行

```bash
colcon build --packages-select racing_segmentation
```

```bash
source install/setup.bash
ros2 launch racing_segmentation racing_segmentation.launch.py
```

## 修改优化后的 Attention 类

文件目录：`ultralytics/nn/modules/block.py`，`Attention`类的`forward`方法替换成以下内容。

```python
class AAttn(nn.Module):
    def forward(self, x):  # RDK X5
        print(f"{x.shape = }")
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.permute(0, 3, 1, 2).contiguous()  # CHW2HWC like
        max_attn = attn.max(dim=1, keepdim=True).values 
        exp_attn = torch.exp(attn - max_attn)
        sum_attn = exp_attn.sum(dim=1, keepdim=True)
        attn = exp_attn / sum_attn
        attn = attn.permute(0, 2, 3, 1).contiguous()  # HWC2CHW like
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
```

## 修改优化后的 Segment 类

文件目录：`ultralytics/nn/modules/head.py`，`Segment`类的`forward`函数替换成以下内容。除了检测部分的6个头外，还有3个`32×(80×80+40×40+20×20)`掩膜系数张量输出头，和一个`32×160×160`的基底，用于合成结果。

```python
class Segment(Detect):
	def forward(self, x):  # RDK X5
    	result = []
    	for i in range(self.nl):
        	result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        	result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        	result.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous())
    	result.append(self.proto(x[0]).permute(0, 2, 3, 1).contiguous())
    	return result
```