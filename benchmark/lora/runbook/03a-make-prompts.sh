#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
# 终端3：没有自备语料时执行；已有prompts.txt则跳过，不覆盖。
python3 - <<'PYTHON'
from pathlib import Path
from transformers import AutoTokenizer
p=Path('/work/mol/evidence/prompts.txt')
assert not p.exists(), '保留现有语料；需要替换时使用新的实验目录'
tok=AutoTokenizer.from_pretrained('/work/mol/base')
tasks=[
    '用两句话解释什么是缓存。', '计算17乘以23，并写出计算步骤。',
    '将“实验结果需要保留原始记录”翻译成英文。',
    '写一个Python函数，返回列表中的最大值。',
    '解释为什么固定输入有助于比较实验。', '给出三个排查服务超时的步骤。',
    '将数字8、3、11、1按从小到大排列。',
    'Explain the difference between latency and throughput.',
    '用JSON列出训练、验证、部署三个阶段。',
    '概括背景材料，并提出一个可以验证的问题。',
]
context='背景材料：团队在测试模型服务。每轮实验使用固定输入，记录配置、原始响应和耗时；发现差异后保留失败样本，修改实现，再使用新的输出目录复测。'
rows=[]
for group,target in [('short',16),('medium',256),('long',2048)]:
    for i,task in enumerate(tasks):
        prefix=f'测试编号{group}-{i:02d}。'
        while len(tok.encode(prefix+task,add_special_tokens=False)) < target:
            prefix+=context
        rows.append(prefix+task)
# 交错排列，前4条短测/混流也包含长输入。
rows=[rows[i+10*g] for i in range(10) for g in range(3)]
lengths=[len(tok.encode(s,add_special_tokens=False)) for s in rows]
assert len(rows)==len(set(rows))==30
assert all(2<=n<=3840 for n in lengths)
with p.open('x') as f:
    f.write('\n'.join(rows)+'\n')
print('prompts=',len(rows),'token_lengths=',lengths)
PYTHON
