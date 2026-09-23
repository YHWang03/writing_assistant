import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.tools.builtin.pdf import ParsePDFTool, ParseAndStoreTool


META = json.dumps(dict(title='A paper', authors='Alice', year=2025, abstract='Summary'))


def pdf(path, text='A paper by Alice'):
    path.write_bytes(b'fixture:' + text.encode())


class FakeDocument:
    needs_pass = False

    def __init__(self, stream, filetype):
        if not stream.startswith(b'fixture:'):
            raise ValueError('corrupt PDF')
        self.text = stream[8:].decode()

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __iter__(self): return iter([SimpleNamespace(get_text=lambda: self.text)])


class PDFCacheTests(unittest.TestCase):
    def setUp(self):
        mocked = patch.dict('sys.modules', {'fitz': SimpleNamespace(open=FakeDocument)})
        mocked.start()
        self.addCleanup(mocked.stop)
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root/'one.pdf'
        pdf(self.path)
        self.tool = ParsePDFTool(self.root/'cache')

    def test_success_survives_restart_and_path_changes(self):
        with patch.object(self.tool, '_extract_with_llm', return_value=META) as extract:
            first = self.tool._parse_one_worker(str(self.path))
            again = self.tool._parse_one_worker(str(self.path))
        self.assertEqual(extract.call_count, 1)
        self.assertFalse(first['cache_hit'])
        self.assertTrue(again['cache_hit'])
        other = self.root/'copy.pdf'
        other.write_bytes(self.path.read_bytes())
        restarted = ParsePDFTool(self.root/'cache')
        with patch.object(restarted, '_extract_with_llm') as extract:
            result = restarted._parse_one_worker(str(other))
        extract.assert_not_called()
        self.assertEqual(result['file'], str(other))

    def test_changed_contents_invalidate_cache(self):
        with patch.object(self.tool, '_extract_with_llm', return_value=META) as extract:
            first = self.tool._parse_one_worker(str(self.path))
            replacement = self.root/'new.pdf'
            pdf(replacement, 'Different content')
            self.path.write_bytes(replacement.read_bytes())
            second = self.tool._parse_one_worker(str(self.path))
        self.assertEqual(extract.call_count, 2)
        self.assertNotEqual(first['fingerprint'], second['fingerprint'])

    def test_parallel_duplicate_content_calls_model_once(self):
        with patch.object(self.tool, '_extract_with_llm', return_value=META) as extract:
            data = json.loads(self.tool.execute(pdf_paths=[str(self.path)]*4))
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(data['success'], 4)
        self.assertEqual(sum(r['cache_hit'] for r in data['results']), 3)

    def test_empty_and_corrupt_files_never_call_model(self):
        empty = self.root/'empty.pdf'
        pdf(empty, '')
        corrupt = self.root/'bad.pdf'
        corrupt.write_bytes(b'not pdf')
        with patch.object(self.tool, '_extract_with_llm') as extract:
            data = json.loads(self.tool.execute(pdf_paths=[str(empty), str(corrupt)]))
        extract.assert_not_called()
        self.assertEqual(data['failed'],2)
        self.assertEqual({r['error_kind'] for r in data['errors']}, {'empty_text','pdf_unreadable'})

    def test_transient_failure_has_bounded_retry_and_session_suppression(self):
        failure = json.dumps(dict(error='temporary',error_kind='transient_api',retryable=True))
        with patch.object(self.tool, '_extract_with_llm', return_value=failure) as extract:
            first = self.tool._parse_one_worker(str(self.path))
            again = self.tool._parse_one_worker(str(self.path))
        self.assertEqual(extract.call_count,2)
        self.assertEqual(first['attempts'],2)
        self.assertTrue(first['retry_exhausted'])
        self.assertTrue(again['cache_hit'])

    def test_invalid_metadata_is_not_cached_as_success(self):
        with patch.object(self.tool, '_extract_with_llm', return_value='[]') as extract:
            first = self.tool._parse_one_worker(str(self.path))
            self.tool._parse_one_worker(str(self.path))
        self.assertEqual(extract.call_count,1)
        self.assertEqual(first['status'],'failed')

    def test_retry_recovers_only_failed_file_and_store_reports_status(self):
        store = ParseAndStoreTool()
        store._parse_tool = self.tool
        refs = []
        store.set_add_func(refs.append)
        transient = json.dumps(dict(error='temporary',retryable=True,error_kind='transient_api'))
        with patch.object(self.tool, '_extract_with_llm', side_effect=[transient,META]) as extract:
            first = json.loads(store.execute(pdf_path=str(self.path)))
            second = json.loads(store.execute(pdf_path=str(self.path)))
        self.assertEqual(extract.call_count,2)
        self.assertEqual(first['stored'],1)
        self.assertEqual(second['cached'],1)
        self.assertEqual(second['files'][0]['attempts'],2)

    def test_mixed_batch_does_not_repeat_successes_or_permanent_failures(self):
        other = self.root/'other.pdf'
        pdf(other, 'Other paper')
        def extract(text, path):
            return META if path == str(self.path) else '{"error":"invalid","retryable":false}'
        with patch.object(self.tool, '_extract_with_llm', side_effect=extract) as call:
            first = json.loads(self.tool.execute(pdf_paths=[str(self.path),str(other)]))
            second = json.loads(self.tool.execute(pdf_paths=[str(self.path),str(other)]))
        self.assertEqual(call.call_count,2)
        self.assertEqual(first['success'],1)
        self.assertEqual(second['failed'],1)
        self.assertTrue(second['results'][0]['cache_hit'])
        self.assertTrue(second['errors'][0]['cache_hit'])

    def test_truncation_does_not_add_outer_retries(self):
        from src.core.llm import IncompleteResponseError
        with patch('src.tools.builtin.pdf.get_tool_llm') as get:
            get.return_value.chat.side_effect = IncompleteResponseError('incomplete')
            result = self.tool._parse_one_worker(str(self.path))
        self.assertEqual(get.return_value.chat.call_count,1)
        self.assertEqual(result['error_kind'],'incomplete_output')


if __name__ == '__main__': unittest.main()
