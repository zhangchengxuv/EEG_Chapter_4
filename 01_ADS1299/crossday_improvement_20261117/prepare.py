from pathlib import Path
p=Path(__file__).parent/'code/training.py'
s=p.read_text()
needle='            optimizer.zero_grad(set_to_none=True)'
s=s.replace(needle,'''            # 仅对训练批次增强；幅度扰动保持目标频率，验证及推理完全不增强。
            if config.get("augment", False):
                gain = torch.empty((len(batch_x), 1, 8, 1), device=batch_x.device).uniform_(0.8, 1.2)
                batch_x = batch_x * gain + torch.randn_like(batch_x) * 0.03
'''+needle)
p.write_text(s)
