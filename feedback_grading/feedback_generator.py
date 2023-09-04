from fuzzywuzzy import fuzz
import ast
import os

"""
Provide feedback
"""



def feedback():
    """
    provide feedback
    :return: error_list, error_keys, fb
    """

    ref_compare = []
    fb = []
    error_keys = []
    with open(os.path.dirname(os.getcwd()) + '/' + 'component_identification/ref_compare.txt', 'r') as f:
        for line in f:
            ref_compare.append([ast.literal_eval(line.strip()), next(f).strip()])
    error_list = []

    # main loop to get the missing and incorrect information
    for i, pair in enumerate(ref_compare):
        incorrect_count = 0
        missing_count = 0
        incorrect_keys = []
        missing_keys = []
        ext_ref = pair[0]
        online_ref = ast.literal_eval(pair[1])
        fb.append(f'for reference {i}:')
        fb.append(ext_ref)
        fb.append(online_ref)
        try:
            online_ref['authors'] = ', '.join(online_ref['authors'])
        except KeyError:
            pass

        for key in ext_ref.keys():
            try:
                score = fuzz.token_set_ratio(ext_ref[key], online_ref[key])
            except KeyError:
                continue
            # print(f'reference {i} score for {key} is {score}')
            if score <= 50:
                fb.append(f'your {key} is incorrect!')
                incorrect_keys.append(key)
                incorrect_count += 1

        for key in online_ref.keys():
            if key not in ext_ref:
                fb.append(f'{key} is missing')
                missing_keys.append(key)
                missing_count += 1
        error_list.append([incorrect_count, missing_count])
        fb.append('\n')
        error_keys.append([incorrect_keys, missing_keys])

    return error_list, error_keys, fb


if __name__ == '__main__':
    print(feedback())

