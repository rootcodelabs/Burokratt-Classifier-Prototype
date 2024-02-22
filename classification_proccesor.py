
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