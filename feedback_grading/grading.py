from feedback_grading.feedback_generator import *

"""
provide grades and summary
"""


def grade():
    """
    weighted grade
    :return: grades, fb, final_summ
    """
    error_list, error_keys, fb = feedback()
    for i in range(2):
        missing_count = [sublist[i] for sublist in error_list]
        incorrect_count = [sublist[i] for sublist in error_list]

    # define weights
    weights = {
        'authors': 0.2,
        'title': 0.2,
        'volume': 0.1,
        'issue': 0.1,
        'pages': 0.1,
        'journal': 0.15,
        'year': 0.1,
        'doi': 0.05
    }

    weighted_incorrect_scores = []
    weighted_missing_scores = []
    for error in error_keys:
        for i in range(2):
            weighted_temp_score = 0
            for item in ['authors', 'title', 'volume', 'issue', 'pages', 'journal', 'year', 'doi']:
                weighted_temp_score += weights[item] * int(item not in error[i])
            if i == 0:
                weighted_incorrect_scores.append(weighted_temp_score)
            else:
                weighted_missing_scores.append(weighted_temp_score)

    avg_incorrect_grade = round(sum(weighted_incorrect_scores) / len(weighted_incorrect_scores), 2)
    avg_missing_grade = round(sum(weighted_missing_scores) / len(weighted_missing_scores), 2)
    avg_grade = (avg_missing_grade + avg_incorrect_grade) / 2

    print(avg_missing_grade)
    print(avg_incorrect_grade)
    print(avg_grade)

    # not weighted
    # avg_missing_count = sum(missing_count) / len(missing_count)
    # avg_incorrect_count = sum(incorrect_count) / len(incorrect_count)

    # avg_missing_grade = round((1 - avg_missing_count / 9) * 100, 2)
    # avg_incorrect_grade = round((1 - avg_incorrect_count / 7) * 100, 2)

    grades = [avg_missing_grade, avg_incorrect_grade, avg_grade]
    final_summ = summ(sum(incorrect_count), sum(missing_count), error_keys, grades)

    return grades, fb, final_summ


def summ(missing_count, incorrect_count, error_keys, grades):
    """
    give summary
    :param missing_count
    :param incorrect_count
    :param error_keys
    :param grades
    :return: total_summ
    """

    incorrect_keys = [sublist[0] for sublist in error_keys]
    missing_keys = [sublist[1] for sublist in error_keys]
    component_list = ['authors', 'title', 'volume', 'issue', 'pages', 'journal', 'year', 'doi']
    incorrect_dict = {}
    missing_dict = {}
    for comp in component_list:
        incorrect_dict[comp] = sum(1 for sublist in incorrect_keys if comp in sublist)
    for comp in component_list:
        missing_dict[comp] = sum(1 for sublist in missing_keys if comp in sublist)
    init_summ = ["\n-----------------------------------------------summary----------------------------------------------------------",
                 f"You have totally missed {missing_count} components and the total incorrect components are {incorrect_count} in all retrieved references."]
    incorrect_summ = ["you have total incorrect counts in all your references:"]
    for item in incorrect_dict.keys():
        incorrect_summ.append(f'{item} : {incorrect_dict[item]}')

    missing_summ = ["\nyou have total missing counts in all references:"]
    for item in missing_dict.keys():
        missing_summ.append(f'{item} : {missing_dict[item]}')

    grade_summ = ["\nYour grades are shown below:"]
    for i, item in enumerate(['incorrect', 'missing', 'final']):
        grade_summ.append(f"averaged {item} grade: {grades[i]}")

    total_summ = init_summ + incorrect_summ + missing_summ + grade_summ + ['-----------------------------------summary end---------------------------------------']

    return total_summ


if __name__ == '__main__':
    _, fb, total_summ = grade()
    for each in fb:
        print(each)
    for each in total_summ:
        print(each)