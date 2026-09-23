import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.domain.paper import Paper
from src.domain.paper_context import PaperContext
from src.domain.library import load_library, save_library
from src.domain.bibliography import parse_bib, merge_bib, render_bib
from src.agents.literature_agent import LiteratureAgent
from src.tools.builtin.tex import WriteFileTool


class BibliographyTests(unittest.TestCase):
    def test_updates_deletions_and_empty_snapshot_persist(self):
        with TemporaryDirectory() as directory:
            save_library(directory, [Paper('a', 'old', 'A', 2000), Paper('b', 'B', 'B', 2001)])
            context = PaperContext(reference_library=load_library(directory))
            context.update_reference('a', {'title': 'new'})
            context.remove_reference('b')
            save_library(directory, context.reference_library, replace=True)
            self.assertEqual([(p.cite_key,p.title) for p in load_library(directory)], [('a','new')])
            save_library(directory, [], replace=True)
            self.assertEqual(load_library(directory), [])

    def test_permissions_and_detached_updates(self):
        context = PaperContext(reference_library=[Paper('a','A','A',2000)])
        denied = context.view({'reference_library'}, set())
        with self.assertRaises(AttributeError): denied.remove_reference('a')
        with self.assertRaises(AttributeError): denied.update_reference('a', {'title':'X'})
        changes = {'bib_fields': {'series':'S'}}
        context.update_reference('a', changes)
        changes['bib_fields']['series'] = 'changed'
        self.assertEqual(context.reference_library[0].bib_fields['series'], 'S')

    def test_migration_preserves_abstract_type_and_extra_fields(self):
        old = Paper('a','Old','A',2000, abstract='Keep this')
        bib = '@book{a, title={New {TTI} title}, author={A and B}, year={2001}, series={Series}, publisher={P}}'
        refs = merge_bib([old], bib)
        self.assertEqual(refs[0].abstract, 'Keep this')
        self.assertEqual(parse_bib(render_bib(refs)), parse_bib(bib))
        self.assertEqual(old.title, 'Old')

    def test_rejects_unsupported_or_broken_input(self):
        for text in ['@article{a,title={broken}', '@string{x="y"}', '@article{a,title="x"}', '@article{a,title={a},title={b}}']:
            with self.assertRaises(ValueError): parse_bib(text)

    def test_export_updates_removes_and_keeps_backup(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            context = PaperContext(library_dir=str(root/'library'))
            agent = LiteratureAgent(SimpleNamespace())
            agent.context = context.view({'reference_library','seed_pdf_paths','ref_pdf_paths','library_dir'}, {'reference_library'})
            agent._sync_context_to_tools()
            self.assertNotIn('WriteFileTool', [t.__name__ for t in agent.tool_types])
            out = root/'references.bib'
            old = '@article{old,title={Old},author={A},year={2000}}'
            out.write_text(old, encoding='utf-8')
            context.add_reference(Paper('new','New','A, B',2002))
            with patch('src.tools.builtin._safe_path._PROJECT_ROOT', root):
                result = agent.require_tool('generate_bib_from_ref_library').execute(str(out))
                denied = WriteFileTool().execute(file_path=str(out), content='overwrite')
            self.assertEqual(json.loads(result)['total'],1)
            self.assertTrue(denied.startswith('Error:'))
            self.assertEqual(parse_bib(out.read_text())[0][1], 'new')
            self.assertEqual(next(root.glob('*.bak')).read_text(),old)
            self.assertEqual(load_library(context.library_dir)[0].cite_key,'new')


if __name__ == '__main__': unittest.main()
