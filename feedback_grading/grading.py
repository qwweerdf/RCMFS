from feedback_grading.feedback_generator import *


def grade():
    error_list, fb = feedback()
    for i in range(2):
        missing_count = [sublist[i] for sublist in error_list]
        incorrect_count = [sublist[i] for sublist in error_list]

    avg_missing_count = sum(missing_count) / len(missing_count)
    avg_incorrect_count = sum(incorrect_count) / len(incorrect_count)

    avg_missing_grade = round((1 - avg_missing_count / 9) * 100, 2)
    avg_incorrect_grade = round((1 - avg_incorrect_count / 7) * 100, 2)
    avg_grade = (avg_missing_grade + avg_incorrect_grade) / 2

    print(avg_missing_grade)
    print(avg_incorrect_grade)
    print(avg_grade)

    return [avg_missing_grade, avg_incorrect_grade, avg_grade], fb


if __name__ == '__main__':
    grade()