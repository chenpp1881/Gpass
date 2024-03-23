import json
import glob


def statistics(glob_rule, save_path=None):
    path_list = glob.glob(glob_rule)
    out_puts = {}
    for path in path_list:
        with open(path, 'r+') as file:
            content = file.read()
        content = json.loads(content)

        for result in content['results']:
            if result['proof_name'] in out_puts:
                out_puts[result['proof_name']] = out_puts[result['proof_name']] | result['success']
            else:
                out_puts[result['proof_name']] = result['success']
    assert len(out_puts) != 0

    if save_path != None:
        dict_json = json.dumps(out_puts)

        with open(rf'{save_path}\result.json', 'w+') as file:
            file.write(dict_json)
    # print(path_list)
    print(sum(out_puts.values()), sum(out_puts.values()) / len(out_puts))
    return out_puts


project = 8
tok = statistics(glob_rule=rf'.\Gpass-master\result\tok\**\results_{project}_*.json')
ours = statistics(glob_rule=rf'.\Gpass-master\result\ours\{project}\*.json')

n = 0
our = 0
toks = 0
for i in ours:
    if ours[i] ^ tok[i]:
        n += (ours[i] ^ tok[i])
        if ours[i]:
            print(f'ours:{i}')
            our += 1
        else:
            print(f'tok:{i}')
            toks +=1
print(n,our,toks)