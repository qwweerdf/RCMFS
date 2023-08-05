from fuzzywuzzy import fuzz
import ast

if __name__ == '__main__':

    ref_compare = []
    with open('ref_compare.txt', 'r') as f:
        for line in f:
            ref_compare.append([ast.literal_eval(line.strip()), next(f).strip()])

    for i, pair in enumerate(ref_compare):
        ext_ref = pair[0]
        online_ref = ast.literal_eval(pair[1])
        print(f'for reference {i}:')
        print(ext_ref)
        print(online_ref)
        online_ref['authors'] = ', '.join(online_ref['authors'])
        for key in ext_ref.keys():
            score = fuzz.token_set_ratio(ext_ref[key], online_ref[key])
            # print(f'reference {i} score for {key} is {score}')
            if score <= 50:
                print(f'your {key} is incorrect!')

        for key in online_ref.keys():
            if key not in ext_ref:
                print(f'{key} is missing')
        print()
