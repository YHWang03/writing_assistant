"""Report generation follows flushed events and remains safe to open offline."""

import json
import logging
from pathlib import Path
import tempfile
import unittest

from src.observability.log_report import read_usage, refresh_reports, render_report
from src.observability.logging_setup import ReportFileHandler, StructuredFormatter


class TestLogReport(unittest.TestCase):
    def test_handler_updates_report_after_summary_and_accumulates_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'structured').mkdir()
            path = root / 'structured/run.jsonl'
            handler = ReportFileHandler(path, encoding='utf-8')
            handler.setFormatter(StructuredFormatter())
            try:
                for task in ['one', 'two']:
                    for event in ['llm_call', 'task_usage']:
                        record = logging.makeLogRecord(dict(
                            msg='message', levelno=logging.INFO, levelname='INFO',
                            event=event, task_id=task, status='success', input_tokens=0,
                            output_tokens=None, usage={'custom': '<script>bad</script>'}))
                        handler.handle(record)
                    report = (root / 'reports/run.html').read_text(encoding='utf-8')
                    self.assertIn('"task_id": "' + task + '"', report)
                self.assertEqual(len(read_usage(path)['events']), 4)
                self.assertIn('reports/run.html', (root / 'index.html').read_text(encoding='utf-8'))
                self.assertNotIn('<script>bad</script>', report)
            finally:
                handler.close()

    def test_legacy_and_corrupt_logs_are_explicit_without_invented_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'structured').mkdir()
            path = root / 'structured/old.jsonl'
            path.write_text('{"event":"agent_step","tokens":123}\ninvalid\n[]\n', encoding='utf-8')
            data = read_usage(path)
            self.assertEqual(data['events'], [])
            self.assertEqual(data['skipped'], 2)
            refresh_reports(root)
            self.assertIn('旧版日志', (root / 'index.html').read_text(encoding='utf-8'))

    def test_json_embedding_roundtrips_and_omits_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.jsonl'
            path.write_text(json.dumps({'event': 'llm_call', 'model': '</script><script>alert(1)</script>',
                                       'msg': 'private prompt', 'input_tokens': None}), encoding='utf-8')
            data = read_usage(path)
            document = render_report(data)
            payload = document.split('id="data">')[1].split('</script>')[0]
            self.assertEqual(json.loads(payload), data)
            self.assertNotIn('private prompt', document)
            self.assertNotIn('<script>alert(1)</script>', document)
            self.assertNotIn('fetch(', document)


if __name__ == '__main__':
    unittest.main()
