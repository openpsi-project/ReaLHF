import json
import numpy as np

for split in ['train', 'valid']:
    filename = f"/data/aigc/llm/checkpoints/fw/senti-genscore-s1/rw-{split}/default@pp_00-mp_00-dp_00/gen_score_data.jsonl"
    with open(filename, 'r') as f:
        data = [json.loads(ff) for ff in f]

    new_data = []
    for x in data:
        pairs = []
        for i, (a1, s1) in enumerate(zip(x['answers'], x['scores'])):
            for j in range(i):
                a2, s2 = x['answers'][j], x['scores'][j]
                if s1 <= s2:
                    s1, s2 = s2, s1
                    a1, a2 = a2, a1
                if s1 > 0.5 and s2 < 0.5:
                    pairs.append((a1, a2))
        if len(pairs) == 0:
            assert all([s < 0.5 for s in x['scores']]) or all([s > 0.5 for s in x['scores']])
            continue
        else:
            assert not all([s < 0.5 for s in x['scores']]) and not all([s > 0.5 for s in x['scores']])
        pos_answers, neg_answers = map(list, zip(*pairs))
        assert len(pos_answers) == len(neg_answers)
        new_data.append(dict(prompt=x['prompt'], pos_answers=pos_answers, neg_answers=neg_answers))

    print(len(new_data), len(data))
    print(np.mean([len(x['pos_answers']) for x in new_data]))

    with open(f"/lustre/fw/datasets/imdb/rl/rm_paired-{split}.jsonl", 'w') as f:
        for x in new_data:
            json.dump(x, f, ensure_ascii=False)
            f.write('\n')