
from io import StringIO
class ClassificationReportParser:
    def __init__(self, report_string):
        self.report_string = report_string

    def parse_report(self):
        report_file = StringIO(self.report_string)
        report_dict = {}
        header_skipped = False
        for line in report_file:
            if line.strip() == '':
                continue
            if not header_skipped:
                header_skipped = True
                continue
            if line.startswith('accuracy'):
                accuracy_line = line.split()
                report_dict['accuracy'] = float(accuracy_line[1])
            else:
                line_parts = line.split()
                if len(line_parts) < 2 or not line_parts[0].isdigit():
                    continue  # Skip lines that do not contain class-wise metrics
                class_label = line_parts[0]
                precision = float(line_parts[1])
                recall = float(line_parts[2])
                f1_score = float(line_parts[3])
                support = int(line_parts[4])
                report_dict[class_label] = {
                    'precision': precision,
                    'recall': recall,
                    'f1-score': f1_score,
                    'support': support
                }
        return report_dict


# # Sample classification report string
# report_string = '''
#               precision    recall  f1-score   support

#            0       0.78      0.88      0.82        67
#            1       0.71      0.68      0.69        41
#            2       0.82      0.68      0.74        37
#            3       0.271      0.648      0.629        451
#            4       0.832      0.648      0.734        347

#     accuracy                           0.77       145
#    macro avg       0.77      0.75      0.75       145
# weighted avg       0.77      0.77      0.77       145
# '''

# # Create an instance of the ClassificationReportParser class
# parser = ClassificationReportParser(report_string)

# # Parse the report
# report_dict = parser.parse_report()

# print(report_dict)